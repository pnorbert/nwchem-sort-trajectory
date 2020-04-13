/*
 * Trajectory test streaming code for imitating NWCHEM with unsorted ADIOS2
 * trajectory data Reads <CASENAME>_trj_nwchem.bp and writes
 * <CASENAME>_trj_dump.bp
 *
 * Norbert Podhorszki, pnorbert@ornl.gov
 *
 */

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <inttypes.h>
#include <iostream>
#include <numeric> //std::accumulate
#include <sstream> // std::ostringstream
#include <stdexcept>
#include <string>
#include <thread>

#include "adios2.h"

MPI_Comm comm;
int rank, comm_size;

size_t sum_sizes(std::vector<std::vector<int64_t>> &arrays)
{
    size_t nelems = 0;
    for (int i = 0; i < arrays.size(); ++i)
        nelems += arrays[i].size();
    return nelems;
}

std::string printDims(const adios2::Dims &dims)
{
    std::ostringstream oss;
    oss << "{";
    size_t n = dims.size();
    if (n > 0)
    {
        oss << dims[0];
        for (size_t i = 1; i < n; ++i)
        {
            oss << ", " << dims[i];
        }
    }
    oss << "}";
    return oss.str();
}

bool epsilon(double d) { return (fabs(d) < 1.0e-20); }
bool epsilon(int64_t d) { return (d == 0); }

/* Gather one array on 'root' process  */
template <class T>
void dbgCheckZeros(bool flag, adios2::Variable<T> &v, std::vector<T> &mydata,
                   std::vector<int64_t> &myindex, int recordsize, int root)
{
    int firstZeroPos = -1;
    if (root == rank)
    {
        const size_t nMyElems = myindex.size() * recordsize;
        assert(mydata.size() == nMyElems);
        int nZeros = 0;
        for (int i = 0; i < nMyElems; ++i)
        {
            if (epsilon(mydata[i]))
            {
                ++nZeros;
                if (firstZeroPos < 0)
                {
                    firstZeroPos = i;
                }
            }
        }
        std::cout << "-- Rank " << rank << " var " << v.Name() << " data has "
                  << nMyElems << " elements and " << nZeros << " zeros ";
        if (firstZeroPos >= 0)
        {
            std::cout << "starting at pos " << firstZeroPos;
        }
        std::cout << std::endl;
    }
}

/*
 * Print info to the user on how to invoke the application
 */
void printUsage()
{
    std::cout << "Usage: nwchem-sort-trajectory CASENAME\n"
              << "  CASENAME:  Name of the nwchem run name\n"
              << "    This tool reads <CASENAME>_trj_dump.bp\n"
              << "    and it writes   <CASENAME>_trj.bp\n\n";
}

int work(std::string &casename)
{

    std::string in_filename(casename + "_trj_nwchem.bp");
    std::string out_filename(casename + "_trj_dump.bp");

    /*
     *   NWCHEM Trajectory data that is needed for sorting
     */

    /* Single time constants */

    // Total Number of Solvent molecules (w as in water)
    int64_t nwm;
    // Number of atoms in each solvent molecule
    int64_t nwa;
    // Total Number of Solute atoms (s as in solute)
    // Total Number of Solute atoms (s as in solute)
    int64_t nsa;

    /* Data changing every step */

    // Number of solvent molecules per process, changing per step
    std::vector<int64_t> nwmn;
    // Number of solute atomes per process, changing per step
    std::vector<int64_t> nsan;

    // Number of writer processes (number of blocks of each array)
    //    this is nproc in input stream, single step constant
    int64_t nwriters;

    // Six logical flags for presence of coords, velocity and forces
    // std::vector<char> flags[6];
    // Flag byte encoding true/false bits
    char flags;

    // actual trajectory data:  coordinates, velocity and forces
    // solvent data is 3D (nwa x 3 x nwmn) !! fast dim is atoms !!
    // each process may read multiple blocks, hence the vector of vector
    std::vector<std::vector<double>> xw, vw, fw;
    // solute data is 2D (3 x nsan)
    std::vector<std::vector<double>> xs, vs, fs;
    // indices (id of solvent molecules and solute atoms)
    std::vector<std::vector<int64_t>> iw, is;

    /*
     *   Other NWCHEM Trajectory data that is passed through
     */
    /* Single time constants */
    std::string rdate;
    /* Data changing every step */
    double stime, pres, temp;
    std::vector<double> vlat;
    std::string rtime;

    /* NWCHEM Output variables (sorted data) only used on rank 0 */

    // solvent data is 3D (nwa x 3 x nwm) !! fast dim is atoms !!
    std::vector<double> oxw, ovw, ofw;
    // solute data is 2D (3 x nsa)
    std::vector<double> oxs, ovs, ofs;

    // Number of blocks to read on this process
    size_t nblocks_solvent, startBlockID_solvent;
    size_t nblocks_solute, startBlockID_solute;

    bool firstStep = true;

    // adios2 variable declarations for some input variables
    adios2::Variable<int64_t> in_vnwmn, in_vnsan;
    adios2::Variable<double> in_vxw, in_vvw, in_vfw, in_vxs, in_vvs, in_vfs;
    adios2::Variable<int64_t> in_viw, in_vis;

    // adios2 variable declarations for the output variables
    adios2::Variable<int64_t> vnwm, vnwa, vnsa, vnproc, vnwmn, vnsan;
    adios2::Variable<double> vxw, vvw, vfw, vxs, vvs, vfs;
    adios2::Variable<std::string> vrdate, vrtime;
    adios2::Variable<double> vstime, vpres, vtemp, vvlat;
    adios2::Variable<int64_t> viw, vis;
    adios2::Variable<int8_t> vflags;

    // adios2 io object and engine init
    adios2::ADIOS ad("adios2.xml", comm);

    // IO objects for reading and writing
    adios2::IO reader_io = ad.DeclareIO("testStreamInput");
    // We use the IO and Variable definitions on all processes but only
    // rank 0 will use it for writing output
    adios2::IO writer_io = ad.DeclareIO("trj");
    if (!rank)
    {
        std::cout << "testStream reads " << in_filename
                  << " using engine type: " << reader_io.EngineType()
                  << std::endl;
        std::cout << "testStream writes " << out_filename
                  << " using engine type:      " << writer_io.EngineType()
                  << std::endl;
    }

    // Engines for reading and writing
    adios2::Engine reader =
        reader_io.Open(in_filename, adios2::Mode::Read, comm);
    adios2::Engine writer;
    writer = writer_io.Open(out_filename, adios2::Mode::Write, comm);

    // read data per timestep
    int stepStream = 0;
    while (true)
    {

        // Begin step
        adios2::StepStatus read_status =
            reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady)
        {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK)
        {
            break;
        }

        int stepSimOut = reader.CurrentStep();

        // Inquire variable and set the selection at the first step only
        // This assumes that the number of atoms do not change in NWCHEM

        if (firstStep)
        {
            // Inquire the scalar variables
            adios2::Variable<int64_t> in_vnwriters =
                reader_io.InquireVariable<int64_t>("nproc");
            adios2::Variable<int64_t> in_vnwm =
                reader_io.InquireVariable<int64_t>("solvent/nwm");
            adios2::Variable<int64_t> in_vnwa =
                reader_io.InquireVariable<int64_t>("solvent/nwa");
            adios2::Variable<int64_t> in_vnsa =
                reader_io.InquireVariable<int64_t>("solute/nsa");

            nwriters = in_vnwriters.Min();
            nwm = in_vnwm.Min();
            nwa = in_vnwa.Min();
            nsa = in_vnsa.Min();

            // Declare variables to output
            // We declare them on every process but only write them on one
            size_t snwriters = static_cast<size_t>(nwriters);
            size_t srank = static_cast<size_t>(rank);
            size_t snwm = static_cast<size_t>(nwm);
            size_t snwa = static_cast<size_t>(nwa);
            viw = writer_io.DefineVariable<int64_t>(
                "solvent/indices", {}, {}, {adios2::UnknownDim}, false);
            vxw = writer_io.DefineVariable<double>(
                "solvent/coords", {}, {}, {3, snwa, adios2::UnknownDim}, false);
            vvw = writer_io.DefineVariable<double>(
                "solvent/velocity", {}, {}, {3, snwa, adios2::UnknownDim});
            vfw = writer_io.DefineVariable<double>(
                "solvent/forces", {}, {}, {3, snwa, adios2::UnknownDim});

            size_t snsa = static_cast<size_t>(nsa);
            vis = writer_io.DefineVariable<int64_t>(
                "solute/indices", {}, {}, {adios2::UnknownDim}, false);
            vxs = writer_io.DefineVariable<double>(
                "solute/coords", {}, {}, {3, adios2::UnknownDim}, false);
            vvs = writer_io.DefineVariable<double>("solute/velocity", {}, {},
                                                   {3, adios2::UnknownDim});
            vfs = writer_io.DefineVariable<double>("solute/forces", {}, {},
                                                   {3, adios2::UnknownDim});

            vnwmn = writer_io.DefineVariable<int64_t>(
                "solvent/nwmn", {snwriters}, {srank}, {1});
            vnsan = writer_io.DefineVariable<int64_t>(
                "solute/nsan", {snwriters}, {srank}, {1});

            if (!rank)
            {
                vnwm = writer_io.DefineVariable<int64_t>("solvent/nwm");
                vnwa = writer_io.DefineVariable<int64_t>("solvent/nwa");
                vnsa = writer_io.DefineVariable<int64_t>("solute/nsa");
                vflags = writer_io.DefineVariable<int8_t>("flags");

                vrdate = writer_io.DefineVariable<std::string>("rdate");
                vrtime = writer_io.DefineVariable<std::string>("rtime");
                vnproc = writer_io.DefineVariable<int64_t>("nproc");

                vstime = writer_io.DefineVariable<double>("stime");
                vpres = writer_io.DefineVariable<double>("pres");
                vtemp = writer_io.DefineVariable<double>("temp");

                vvlat = writer_io.DefineVariable<double>("vlat", {3, 3}, {0, 0},
                                                         {3, 3});
            }
        }

        // How many blocks to read on this process?
        size_t total_solvent_blocks;
        size_t total_solute_blocks;
        {
            // Total number of solvent blocks in this step
            in_viw = reader_io.InquireVariable<int64_t>("solvent/indices");
            auto blocks = reader.BlocksInfo(in_viw, reader.CurrentStep());
            total_solvent_blocks = blocks.size();
            std::cout << "   ****  Solvent blocks = " << total_solvent_blocks
                      << "   ****" << std::endl;
        }
        {
            // Total number of solute blocks in this step
            in_vis = reader_io.InquireVariable<int64_t>("solute/indices");
            auto blocks = reader.BlocksInfo(in_vis, reader.CurrentStep());
            total_solute_blocks = blocks.size();
            std::cout << "   ****  Solute blocks = " << total_solute_blocks
                      << "   ****" << std::endl;
        }

        {
            // Number of solvent blocks to read by this process in this step
            nblocks_solvent =
                static_cast<size_t>(total_solvent_blocks / comm_size);
            size_t extras =
                static_cast<size_t>(total_solvent_blocks % comm_size);
            startBlockID_solvent = rank * nblocks_solvent;
            if (rank < extras)
            {
                ++nblocks_solvent;
                startBlockID_solvent += rank;
            }
            else
            {
                startBlockID_solvent += extras;
            }
        }

        {
            // Number of solute blocks to read by this process in this step
            nblocks_solute =
                static_cast<size_t>(total_solute_blocks / comm_size);
            size_t extras =
                static_cast<size_t>(total_solute_blocks % comm_size);
            startBlockID_solute = rank * nblocks_solute;
            if (rank < extras)
            {
                ++nblocks_solute;
                startBlockID_solute += rank;
            }
            else
            {
                startBlockID_solute += extras;
            }
        }

        std::cout << "Rank " << rank << " reads " << nblocks_solvent
                  << "  blocks from idx " << startBlockID_solvent
                  << " from total " << total_solvent_blocks
                  << " Solvent blocks, and reads " << nblocks_solute
                  << "  blocks from idx " << startBlockID_solute
                  << " from total " << total_solute_blocks << " Solute blocks"
                  << std::endl;

        iw.resize(nblocks_solvent);
        xw.resize(nblocks_solvent);
        vw.resize(nblocks_solvent);
        fw.resize(nblocks_solvent);
        is.resize(nblocks_solute);
        xs.resize(nblocks_solute);
        vs.resize(nblocks_solute);
        fs.resize(nblocks_solute);

        // process flags
        adios2::Variable<int8_t> in_vflags =
            reader_io.InquireVariable<int8_t>("flags");
        flags = static_cast<char>(in_vflags.Min());
        bool lfs = (flags % 2 == 1);
        flags = flags >> 1;
        bool lvs = (flags % 2 == 1);
        flags = flags >> 1;
        bool lxs = (flags % 2 == 1);
        flags = flags >> 1;
        bool lfw = (flags % 2 == 1);
        flags = flags >> 1;
        bool lvw = (flags % 2 == 1);
        flags = flags >> 1;
        bool lxw = (flags % 2 == 1);
        flags = flags >> 1;
        std::cout << "Rank " << rank << " flags :" << lxw << lvw << lfw << lxs
                  << lvs << lfs << std::endl;

        // Read trajectory data and indices
        if (lxw)
            in_vxw = reader_io.InquireVariable<double>("solvent/coords");
        if (lvw)
            in_vvw = reader_io.InquireVariable<double>("solvent/velocity");
        if (lfw)
            in_vfw = reader_io.InquireVariable<double>("solvent/forces");

        if (lxs)
            in_vxs = reader_io.InquireVariable<double>("solute/coords");
        if (lvs)
            in_vvs = reader_io.InquireVariable<double>("solute/velocity");
        if (lfs)
            in_vfs = reader_io.InquireVariable<double>("solute/forces");

        for (size_t i = 0; i < nblocks_solvent; ++i)
        {
            std::cout << "Rank " << rank << " solvent block " << i
                      << " select ID " << startBlockID_solvent + i << std::endl;
            in_viw.SetBlockSelection(startBlockID_solvent + i);
            reader.Get<int64_t>(in_viw, iw[i]);
            std::cout << "Rank " << rank << " block " << i << " solvent/indices"
                      << " shape = " << printDims(in_viw.Shape())
                      << " dims = " << printDims(in_viw.Count())
                      << " offset = " << printDims(in_viw.Start()) << std::endl;
            if (lxw)
            {
                in_vxw.SetBlockSelection(startBlockID_solvent + i);
                size_t n = 1;
                for (const auto &d : in_vxw.Count())
                {
                    n *= d;
                }
                /*xw[i].resize(n);*/
                reader.Get<double>("solvent/coords", xw[i]);
                std::cout << "Rank " << rank << " block " << i
                          << " solvent/coords size = " << n
                          << " shape = " << printDims(in_vxw.Shape())
                          << " dims = " << printDims(in_vxw.Count())
                          << " offset = " << printDims(in_vxw.Start())
                          << std::endl;
            }
            if (lvw)
            {
                in_vvw.SetBlockSelection(startBlockID_solvent + i);
                reader.Get<double>(in_vvw, vw[i]);
            }
            if (lfw)
            {
                in_vfw.SetBlockSelection(startBlockID_solvent + i);
                reader.Get<double>(in_vfw, fw[i]);
            }
        }

        for (size_t i = 0; i < nblocks_solute; ++i)
        {
            std::cout << "Rank " << rank << " solute block " << i
                      << " select ID " << startBlockID_solvent + i << std::endl;
            in_vis.SetBlockSelection(startBlockID_solute + i);
            reader.Get<int64_t>(in_vis, is[i]);
            if (lxs)
            {
                in_vxs.SetBlockSelection(startBlockID_solute + i);
                reader.Get<double>(in_vxs, xs[i]);
            }
            if (lvs)
            {
                in_vvs.SetBlockSelection(startBlockID_solute + i);
                reader.Get<double>(in_vvs, vs[i]);
            }
            if (lfs)
            {
                in_vfs.SetBlockSelection(startBlockID_solute + i);
                reader.Get<double>(in_vfs, fs[i]);
            }
        }

        // Read rest of information
        reader.Get<int64_t>("solvent/nwmn", nwmn);
        reader.Get<int64_t>("solute/nsan", nsan);
        if (!rank)
        {
            if (firstStep)
            {
                reader.Get<std::string>("rdate", rdate);
            }
            reader.Get<std::string>("rtime", rtime);
            reader.Get<double>("stime", stime);
            reader.Get<double>("pres", pres);
            reader.Get<double>("temp", temp);
            reader.Get<double>("vlat", vlat);
        }

        // End adios2 step for reading
        reader.EndStep();

        // Write out result

        writer.BeginStep();

        if (!rank)
        {
            if (firstStep)
            {
                writer.Put<int64_t>(vnwm, nwm);
                writer.Put<int64_t>(vnwa, nwa);
                writer.Put<int64_t>(vnsa, nsa);
                writer.Put<int64_t>(vnproc, nwriters);
                writer.Put<std::string>(vrdate, rdate);
            }
            writer.Put<double>(vstime, stime);
            writer.Put<double>(vpres, pres);
            writer.Put<double>(vtemp, temp);
            writer.Put<std::string>(vrtime, rtime);
            writer.Put<double>(vvlat, vlat.data());
            int8_t f = in_vflags.Min();
            writer.Put<int8_t>(vflags, f);
        }

        for (size_t i = 0; i < nblocks_solvent; ++i)
        {
            vnwmn.SetSelection({{startBlockID_solvent + i}, {1}});
            writer.Put<int64_t>(vnwmn, nwmn[startBlockID_solvent + i],
                                adios2::Mode::Sync);

            in_viw.SetBlockSelection(startBlockID_solvent + i);
            size_t snwm = in_viw.Count()[0];
            size_t snwa = static_cast<size_t>(nwa);
            viw.SetSelection({{}, {snwm}});
            writer.Put<int64_t>(viw, iw[i].data(), adios2::Mode::Sync);
            if (lxw)
            {

                vxw.SetSelection({{}, {3, snwa, snwm}});
                writer.Put<double>(vxw, xw[i].data(), adios2::Mode::Sync);
            }
            if (lvw)
            {
                vvw.SetSelection({{}, {3, snwa, snwm}});
                writer.Put<double>(vvw, vw[i].data());
            }
            if (lfw)
            {
                vfw.SetSelection({{}, {3, snwa, snwm}});
                writer.Put<double>(vfw, fw[i].data());
            }
        }
        for (size_t i = 0; i < nblocks_solute; ++i)
        {
            vnsan.SetSelection({{startBlockID_solute + i}, {1}});
            writer.Put<int64_t>(vnsan, nsan[startBlockID_solute + i],
                                adios2::Mode::Sync);

            in_vis.SetBlockSelection(startBlockID_solute + i);
            size_t snas = in_vis.Count()[0];
            vis.SetSelection({{}, {snas}});
            writer.Put<int64_t>(vis, is[i].data(), adios2::Mode::Sync);
            if (lxs)
            {
                vxs.SetSelection({{}, {3, snas}});
                writer.Put<double>(vxs, xs[i].data());
            }
            if (lvs)
            {
                vvs.SetSelection({{}, {3, snas}});
                writer.Put<double>(vvs, vs[i].data());
            }
            if (lfs)
            {
                vfs.SetSelection({{}, {3, snas}});
                writer.Put<double>(vfs, fs[i].data());
            }
        }
        writer.EndStep();
        ++stepStream;
        firstStep = false;
    }

    // cleanup
    reader.Close();
    writer.Close();

    return 0;
}

/*
 * MAIN
 */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int wrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 3254;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (comm_size < 1)
    {
        if (rank == 0)
        {
            std::cerr
                << "ERROR: This application must run on at least 6 processes\n";
        }
    }
    else if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "ERROR: Not enough arguments\n";
            printUsage();
        }
    }
    else
    {
        std::string casename(argv[1]);
        work(casename);
    }
    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}
