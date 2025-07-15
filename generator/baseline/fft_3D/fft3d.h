#include "test_fft3d.h"

using gpu_backend = heffte::backend::cufft;

template <typename backend_tag, typename index>
void benchmark_fft(std::array<int, 3> size_fft, std::deque<std::string> const &args, MPI_Comm comm)
{
    int me, nprocs;
    MPI_Comm fft_comm = comm;
    MPI_Comm_rank(fft_comm, &me);
    MPI_Comm_size(fft_comm, &nprocs);

    heffte::plan_options options = args_to_options<backend_tag>(args);

    // Create input and output boxes on local processor
    box3d<index> const world = {{0, 0, 0}, {size_fft[0] - 1, size_fft[1] - 1, size_fft[2] - 1}};

    // Get grid of processors at input and output
    std::array<int, 3> proc_i, proc_o;

    if (options.use_pencils == true)
    {
        std::array<int, 2> proc_grid = make_procgrid(nprocs);
        proc_i = {1, proc_grid[0], proc_grid[1]};
        proc_o = {1, proc_grid[0], proc_grid[1]};
    }
    else
    {
        proc_i = {1, 1, nprocs};
        proc_o = {1, 1, nprocs};
    }

    std::vector<box3d<index>> inboxes = heffte::split_world(world, proc_i);
    std::vector<box3d<index>> outboxes = heffte::split_world(world, proc_o);

    if (std::is_same<backend_tag, gpu_backend>::value and has_mps(args))
    {
        heffte::gpu::device_set(me % heffte::gpu::device_count());
    }

    auto fft = fft3d<backend_tag>(inboxes[me], outboxes[me], fft_comm, options);

    std::vector<std::complex<double>> input(fft.size_inbox());
    for (int i = 0; i < fft.size_inbox(); ++i)
    {
        double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        std::complex<double> value(real_part, imag_part);
        input[i] = value;
    }

    std::vector<std::complex<double>> output(std::max(fft.size_outbox(), fft.size_inbox()));
    std::copy(input.begin(), input.end(), output.begin());

    std::complex<double> *output_array = output.data();
    gpu::vector<std::complex<double>> gpu_output;
    if (std::is_same<backend_tag, gpu_backend>::value)
    {
        gpu_output = gpu::transfer::load(output);
        output_array = gpu_output.data();
    }

    // Define workspace array
    typename heffte::fft3d<backend_tag>::template buffer_container<std::complex<double>> workspace(fft.size_workspace());

    // Warmup
    heffte::add_trace("mark warmup begin");
    fft.forward(output_array, output_array, scale::full);
    fft.backward(output_array, output_array);

    // Execution
    int const ntest = 10;
    MPI_Barrier(fft_comm);
    double fft_direct = 0;
    double fft_inverse = 0;

    for (int i = 0; i < ntest; ++i)
    {
        double t0 = MPI_Wtime();

        heffte::add_trace("mark forward begin");
        fft.forward(output_array, output_array, workspace.data(), scale::full);
        if (backend::uses_gpu<backend_tag>::value)
            gpu::synchronize_default_stream();

        double t1 = MPI_Wtime();

        heffte::add_trace("mark backward begin");
        fft.backward(output_array, output_array, workspace.data());
        if (backend::uses_gpu<backend_tag>::value)
            gpu::synchronize_default_stream();

        double t2 = MPI_Wtime();

        MPI_Barrier(fft_comm);

        fft_direct += (t1 - t0);
        fft_inverse += (t2 - t1);
    }

    // Get execution time
    double t_direct_max = 0.0, t_inverse_max = 0.0;
    MPI_Reduce(&fft_direct, &t_direct_max, 1, MPI_DOUBLE, MPI_MAX, 0, fft_comm);
    MPI_Reduce(&fft_inverse, &t_inverse_max, 1, MPI_DOUBLE, MPI_MAX, 0, fft_comm);

    // Validate result
    if (std::is_same<backend_tag, gpu_backend>::value)
    {
        output = gpu::transfer::unload(gpu_output);
    }
    output.resize(input.size()); // match the size of the original input

    double err = 0.0;
    for (size_t i = 0; i < input.size(); i++)
        err = std::max(err, std::abs(input[i] - output[i]));
    double mpi_max_err = 0.0;
    MPI_Allreduce(&err, &mpi_max_err, 1, mpi::type_from<double>(), MPI_MAX, fft_comm);

    if (mpi_max_err > precision<std::complex<double>>::tolerance)
    {
        // benchmark failed, the error is too much
        if (me == 0)
        {
            cout << "------------------------------- \n"
                 << "ERROR: observed error after heFFTe benchmark exceeds the tolerance\n"
                 << "       tolerance: " << precision<std::complex<double>>::tolerance
                 << "  error: " << mpi_max_err << endl;
        }
        return;
    }

    // Print results
    if (me == 0)
    {
        if (options.use_pencils == true)
            cout << size_fft[0] << "x" << size_fft[1] << "x" << batch << "\t" << proc_o[0] << "x" << proc_o[1] << "\tFFT direct [s]:\t" << t_direct_max / ntest << "\tFFT inverse [s]:\t" << t_inverse_max / ntest << endl;
        else
            cout << size_fft[0] << "x" << size_fft[1] << "x" << batch << "\t" << nprocs << "\tFFT direct [s]:\t" << t_direct_max / ntest << "\tFFT inverse [s]:\t" << t_inverse_max / ntest << endl;
    }
}

template <typename backend_tag>
bool perform_benchmark(std::array<int, 3> size_fft, std::deque<std::string> const &args, MPI_Comm comm)
{
    benchmark_fft<backend_tag, int>(size_fft, args, comm);
    return true;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    std::string bench_executable = "./fft3d.x";

    if (argc < 4)
    {
        if (mpi::world_rank(0))
        {
            cout << "\nUsage:\n    mpirun -np x " << bench_executable << " <size-x> <size-y> <size-z> <args>\n\n"
                 << "    options\n"
                 << "        size-x/y/z are the 3D array dimensions \n\n"
                 << "        args is a set of optional arguments that define algorithmic tweaks and variations\n"
                 << "         -reorder: reorder the elements of the arrays so that each 1-D FFT will use contiguous data\n"
                 << "         -no-reorder: some of the 1-D will be strided (non contiguous)\n"
                 << "         -a2a: use MPI_Alltoallv() communication method\n"
                 << "         -p2p: use MPI_Send() and MPI_Irecv() communication methods\n"
                 << "         -pencils: use pencil reshape logic\n"
                 << "         -slabs: use slab reshape logic\n"
                 << "         -io_pencils: if input and output proc grids are pencils, useful for comparison with other libraries \n"
                 << "         -mps: for the cufft backend and multiple gpus, associate the mpi ranks with different cuda devices\n"
                 << "Examples:\n"
                 << "    mpirun -np  4 " << bench_executable << " 128 128 128 -no-reorder\n"
                 << "    mpirun -np  8 " << bench_executable << " 256 256 128\n"
                 << "    mpirun -np 12 " << bench_executable << " 512 512 128 -p2p\n\n";
        }

        MPI_Finalize();
        return 0;
    }

    std::array<int, 3> size_fft = {0, 0, 0};

    try
    {
        size_fft = {std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])};
        for (auto s : size_fft)
            if (s < 1)
                throw std::invalid_argument("negative input");
    }
    catch (std::invalid_argument &e)
    {
        if (mpi::world_rank(0))
        {
            std::cout << "Cannot convert the sizes into positive integers!\n";
            std::cout << "Encountered error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    init_tracing(bench_executable + "_cuda_double_" + std::string(argv[1]) + "_" + std::string(argv[2]) + "_" + std::string(argv[3]));

    bool valid_backend = false;
    valid_backend = valid_backend or perform_benchmark<backend::cufft>(size_fft, arguments(argc, argv), MPI_COMM_WORLD);

    if (not valid_backend)
    {
        if (mpi::world_rank(0))
        {
            std::cout << "Invalid backend cuda\n";
        }

        MPI_Finalize();
        return 0;
    }

    finalize_tracing();

    MPI_Finalize();
    return 0;
}
