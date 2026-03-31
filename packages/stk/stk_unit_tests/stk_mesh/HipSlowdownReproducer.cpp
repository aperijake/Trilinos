// Standalone STK/Kokkos progressive slowdown reproducer for AMD MI300X
//
// This test uses ONLY STK and Kokkos APIs to reproduce a progressive
// slowdown observed on AMD MI300X (SR-IOV, xnack-) with HIPManagedSpace.
//
// ROOT CAUSE FINDING: The slowdown is triggered by sidesets in the mesh.
// A mesh created with "generated:1x1x1|sideset:xXyYzZ|tets" shows progressive
// slowdown, while "generated:1x1x1|tets" (no sidesets) does not.
//
// The slowdown occurs because STK's DeviceBucketRepository uses UVMMemSpace
// (= Kokkos::SharedSpace = HIPManagedSpace on HIP). On SR-IOV virtualized
// MI300X instances with xnack disabled, there is no hardware page migration
// for managed memory. Each mesh create/destroy cycle accumulates SVM range
// tracking overhead in the kernel driver, causing host-side dispatch gaps
// to grow linearly. Sidesets create additional mesh parts and buckets,
// amplifying the managed memory churn.
//
// Build:
//   Build target stk_hip_slowdown_reproducer (see CMakeLists.txt)
//
// Run:
//   ./stk_hip_slowdown_reproducer.exe --gtest_repeat=60
//   ./stk_hip_slowdown_reproducer.exe --gtest_repeat=60 --gtest_filter=*WithSidesets*
//   ./stk_hip_slowdown_reproducer.exe --gtest_repeat=60 --gtest_filter=*NoSidesets*
//
// Expected: ~constant time per iteration
// Observed on MI300X with sidesets: linear increase (~1ms per create/destroy cycle)

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <stk_io/FillMesh.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/NgpFieldBLAS.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>

#include <chrono>
#include <iostream>
#include <string>

class StkHipSlowdownReproducer : public ::testing::Test {
   protected:
    void SetUp() override {
        m_comm = MPI_COMM_WORLD;
        MPI_Comm_size(m_comm, &m_num_procs);
    }

    // Core reproduce cycle: create mesh, register fields, populate,
    // sync to device, run time-step kernels, destroy everything.
    void RunCycle(const std::string& mesh_string) {
        // 1. Create BulkData
        stk::mesh::MeshBuilder builder(m_comm);
        builder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
        std::shared_ptr<stk::mesh::BulkData> bulk = builder.create();
        stk::mesh::MetaData& meta = bulk->mesh_meta_data();

        // 2. Register fields (before populate)
        auto& disp_field = meta.declare_field<double>(
            stk::topology::NODE_RANK, "displacement", 2);
        auto& vel_field = meta.declare_field<double>(
            stk::topology::NODE_RANK, "velocity", 2);
        auto& accel_field = meta.declare_field<double>(
            stk::topology::NODE_RANK, "acceleration", 2);
        auto& force_field = meta.declare_field<double>(
            stk::topology::NODE_RANK, "force", 1);
        auto& mass_field = meta.declare_field<double>(
            stk::topology::NODE_RANK, "mass", 1);
        auto& stress_field = meta.declare_field<double>(
            stk::topology::ELEMENT_RANK, "stress", 2);

        stk::mesh::Selector universal = meta.universal_part();
        double zero3[3] = {0.0, 0.0, 0.0};
        double zero9[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        stk::mesh::put_field_on_mesh(disp_field, universal, 3, zero3);
        stk::mesh::put_field_on_mesh(vel_field, universal, 3, zero3);
        stk::mesh::put_field_on_mesh(accel_field, universal, 3, zero3);
        stk::mesh::put_field_on_mesh(force_field, universal, 3, zero3);
        stk::mesh::put_field_on_mesh(mass_field, universal, 3, zero3);
        stk::mesh::put_field_on_mesh(stress_field, universal, 9, zero9);

        // 3. Populate mesh
        stk::io::fill_mesh(mesh_string, *bulk);

        // 4. Sync to device
        stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(*bulk);
        auto ngp_disp = stk::mesh::get_updated_ngp_field<double>(disp_field);
        auto ngp_vel = stk::mesh::get_updated_ngp_field<double>(vel_field);
        auto ngp_accel = stk::mesh::get_updated_ngp_field<double>(accel_field);
        auto ngp_force = stk::mesh::get_updated_ngp_field<double>(force_field);
        auto ngp_mass = stk::mesh::get_updated_ngp_field<double>(mass_field);
        auto ngp_stress = stk::mesh::get_updated_ngp_field<double>(stress_field);

        // 5. Run 10 "time steps" with field operations
        stk::mesh::Selector active = meta.universal_part();
        for (int step = 0; step < 10; ++step) {
            // Zero force field
            stk::mesh::field_fill(0.0, force_field, active, stk::ngp::ExecSpace());

            // Node loop: trivial velocity/displacement update
            stk::mesh::for_each_entity_run(
                ngp_mesh, stk::topology::NODE_RANK, active,
                KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& node) {
                    for (int i = 0; i < 3; ++i) {
                        double a = ngp_accel(node, i);
                        double v = ngp_vel(node, i);
                        ngp_vel(node, i) = v + 0.1 * a;
                        ngp_disp(node, i) = ngp_disp(node, i) + 0.1 * v;
                        ngp_force(node, i) = ngp_force(node, i) - ngp_mass(node, i) * a;
                    }
                });

            // Element loop: trivial stress update
            stk::mesh::for_each_entity_run(
                ngp_mesh, stk::topology::ELEMENT_RANK, active,
                KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& elem) {
                    for (int i = 0; i < 9; ++i) {
                        ngp_stress(elem, i) = ngp_stress(elem, i) * 0.99;
                    }
                });

            // Mark fields modified on device
            ngp_disp.clear_sync_state();
            ngp_disp.modify_on_device();
            ngp_vel.clear_sync_state();
            ngp_vel.modify_on_device();
            ngp_force.clear_sync_state();
            ngp_force.modify_on_device();

            // Field state rotation
            bulk->update_field_data_states();
            ngp_mesh = stk::mesh::get_updated_ngp_mesh(*bulk);
            ngp_disp = stk::mesh::get_updated_ngp_field<double>(disp_field);
            ngp_vel = stk::mesh::get_updated_ngp_field<double>(vel_field);
            ngp_accel = stk::mesh::get_updated_ngp_field<double>(accel_field);
            ngp_force = stk::mesh::get_updated_ngp_field<double>(force_field);
            ngp_stress = stk::mesh::get_updated_ngp_field<double>(stress_field);
        }

        // 6. Destroy everything (BulkData goes out of scope)
    }

    MPI_Comm m_comm;
    int m_num_procs;
};

// REPRODUCES SLOWDOWN: mesh with sidesets
TEST_F(StkHipSlowdownReproducer, WithSidesets) {
    if (m_num_procs != 1) GTEST_SKIP_("Single process only");
    auto start = std::chrono::high_resolution_clock::now();
    RunCycle("generated:1x1x1|sideset:xXyYzZ|tets");
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] WithSidesets elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
}

// CONTROL: mesh without sidesets — should NOT show slowdown
TEST_F(StkHipSlowdownReproducer, NoSidesets) {
    if (m_num_procs != 1) GTEST_SKIP_("Single process only");
    auto start = std::chrono::high_resolution_clock::now();
    RunCycle("generated:1x1x1|tets");
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] NoSidesets elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
}

// MINIMAL: sidesets but NO field ops — test if mesh lifecycle alone triggers it
TEST_F(StkHipSlowdownReproducer, SidesetsNoFieldOps) {
    if (m_num_procs != 1) GTEST_SKIP_("Single process only");
    auto start = std::chrono::high_resolution_clock::now();

    // Just create mesh with sidesets, sync to device, and destroy
    stk::mesh::MeshBuilder builder(m_comm);
    builder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
    std::shared_ptr<stk::mesh::BulkData> bulk = builder.create();
    stk::io::fill_mesh("generated:1x1x1|sideset:xXyYzZ|tets", *bulk);
    stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(*bulk);
    // Destroy
    bulk.reset();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] SidesetsNoFieldOps elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
