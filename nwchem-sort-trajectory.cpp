/*
 * Trajectory sorting code for NWCHEM unsorted ADIOS2 trajectory data
 * Reads <CASENAME>_trj_dump.bp and writes <CASENAME>_trj.bp
 *
 * Norbert Podhorszki, pnorbert@ornl.gov
 *
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "adios2.h"


MPI_Comm comm;
int rank, comm_size;

/*
 * Function to compute the PDF of a 2D slice
 */
template <class T>
void compute_pdf(const std::vector<T> &data,
                 const std::vector<std::size_t> &shape, const size_t start,
                 const size_t count, const size_t nbins, const T min,
                 const T max, std::vector<T> &pdf, std::vector<T> &bins)
{
    if (shape.size() != 3)
        throw std::invalid_argument("ERROR: shape is expected to be 3D\n");

    size_t slice_size = shape[1] * shape[2];
    pdf.resize(count * nbins);
    bins.resize(nbins);

    size_t start_data = 0;
    size_t start_pdf = 0;

    T binWidth = (max - min) / nbins;
    for (auto i = 0; i < nbins; ++i) {
        bins[i] = min + (i * binWidth);
    }

    if (nbins == 1) {
        // special case: only one bin
        for (auto i = 0; i < count; ++i) {
            pdf[i] = slice_size;
        }
        return;
    }

    if (epsilon(max - min) || epsilon(binWidth)) {
        // special case: constant array
        for (auto i = 0; i < count; ++i) {
            pdf[i * nbins + (nbins / 2)] = slice_size;
        }
        return;
    }

    for (auto i = 0; i < count; ++i) {
        // Calculate a PDF for 'nbins' bins for values between 'min' and 'max'
        // from data[ start_data .. start_data+slice_size-1 ]
        // into pdf[ start_pdf .. start_pdf+nbins-1 ]
        for (auto j = 0; j < slice_size; ++j) {
            if (data[start_data + j] > max || data[start_data + j] < min) {
                std::cout << " data[" << start * slice_size + start_data + j
                          << "] = " << data[start_data + j]
                          << " is out of [min,max] = [" << min << "," << max
                          << "]" << std::endl;
            }
            size_t bin = static_cast<size_t>(
                std::floor((data[start_data + j] - min) / binWidth));
            if (bin == nbins) {
                bin = nbins - 1;
            }
            ++pdf[start_pdf + bin];
        }
        start_pdf += nbins;
        start_data += slice_size;
    }
    return;
}

/*
 * Print info to the user on how to invoke the application
 */
void printUsage()
{
    std::cout
        << "Usage: nwchem-sort-trajectory CASENAME\n"
        << "  CASENAME:  Name of the nwchem run name\n"
        << "    This tool reads <CASENAME>_trj_dump.bp\n"
        << "    and it writes   <CASENAME>_trj.bp\n\n";
}

int work(std::string &casename)
{

    std::string in_filename(casename + "_trj_dump.bp");
    std::string out_filename(casename + "_trj.bp"); 

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
    std::vector<std::int64_t> nsan;

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
    std::vector<std::vector<std::int64_t>> iw, is;

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
    size_t  nblocks, startBlockID; 


    bool firstStep = true;

    // adios2 variable declarations for some input variables
    adios2::Variable<int64_t> in_vnwmn, in_vnsan;
    adios2::Variable<double> in_vxw, in_vvw, in_vfw, in_vxs, in_vvs, in_vfs;
    adios2::Variable<int64_t> in_viw, in_vis;

    // adios2 variable declarations for the output variables
    adios2::Variable<int64_t> vnwm, vnwa, vnsa;
    adios2::Variable<double> vxw, vvw, vfw, vxs, vvs, vfs;
    adios2::Variable<int64_t> viw, vis;
    adios2::Variable<std::string> vrdate, vrtime;
    adios2::Variable<double> vstime, vpres, vtemp, vvlat;

    // adios2 io object and engine init
    adios2::ADIOS ad("adios2.xml", comm, adios2::DebugON);

    // IO objects for reading and writing
    adios2::IO reader_io = ad.DeclareIO("trj");
    adios2::IO writer_io;
    if (!rank) {
        writer_io = ad.DeclareIO("SortingOutput");
        std::cout << "Sorter reads " << in_filename 
                  << " using engine type: "
                  << reader_io.EngineType() << std::endl;
        std::cout << "Sorter writes " << out_filename 
                  << " using engine type:      "
                  << writer_io.EngineType() << std::endl;
    }

    // Engines for reading and writing
    adios2::Engine reader =
        reader_io.Open(in_filename, adios2::Mode::Read, comm);
    adios2::Engine writer;
    if (!rank)
    {
        writer = writer_io.Open(out_filename, adios2::Mode::Write, MPI_COMM_SELF);
    }


    // read data per timestep
    int stepSorting = 0;
    while (true) {

        // Begin step
        adios2::StepStatus read_status =
            reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        } else if (read_status != adios2::StepStatus::OK) {
            break;
        }

        int stepSimOut = reader.CurrentStep();

        // Inquire variable and set the selection at the first step only
        // This assumes that the number of atoms do not change in NWCHEM

        if (firstStep) {
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
            if (!rank)
            {
                size_t snwm =  static_cast<size_t>(nwm);
                size_t snwa =  static_cast<size_t>(nwa);
                vxw = writer_io.DefineVariable<double>(
                        "solvent/coords", {3,snwa,snwm}, {0,0,0}, {3,snwa,snwm});
                vvw = writer_io.DefineVariable<double>(
                        "solvent/velocity", {3,snwa,snwm}, {0,0,0}, {3,snwa,snwm});
                vfw = writer_io.DefineVariable<double>(
                        "solvent/forces", {3,snwa,snwm}, {0,0,0}, {3,snwa,snwm});

                size_t snsa =  static_cast<size_t>(nsa);
                vxs = writer_io.DefineVariable<double>(
                        "solute/coords", {3,snsa}, {0,0}, {3,snsa});
                vvs = writer_io.DefineVariable<double>(
                        "solute/velocity", {3,snsa}, {0,0}, {3,snsa});
                vfs = writer_io.DefineVariable<double>(
                        "solute/forces", {3,snsa}, {0,0}, {3,snsa});

                vnwm = writer_io.DefineVariable<int64_t>("solvent/nwm");
                vnwa = writer_io.DefineVariable<int64_t>("solvent/nwa");
                vnsa = writer_io.DefineVariable<int64_t>("solute/nsa");

                vrdate = writer_io.DefineVariable<std::string>("rdate");
                vrtime = writer_io.DefineVariable<std::string>("rtime");

                vstime = writer_io.DefineVariable<double>("stime");
                vpres = writer_io.DefineVariable<double>("pres");
                vtemp = writer_io.DefineVariable<double>("temp");

                vvlat = writer_io.DefineVariable<double>(
                        "vlat", {3,3}, {0,0}, {3,3});

            }

            // How many blocks to read on this process?
            nblocks = static_cast<size_t>(nwriters / comm_size);
            size_t extras = static_cast<size_t>(nwriters % comm_size);
            startBlockID = rank * nblocks;
            if (rank < extras) 
            {
                ++nblocks;
                startBlockID += rank;
            }
            else
            {
                startBlockID += extras;
            }

            std::cout << "Rank " << rank << " reads " << nblocks 
                      << " blocks from idx " << startBlockID 
                      << " from total " << nwriters << " blocks" << std::endl;
            
            if (!rank)
            {
                iw.resize(nwriters);
                xw.resize(nwriters);
                vw.resize(nwriters);
                fw.resize(nwriters);
                is.resize(nwriters);
                xs.resize(nwriters);
                vs.resize(nwriters);
                fs.resize(nwriters);
            }
            else
            {
                iw.resize(nblocks);
                xw.resize(nblocks);
                vw.resize(nblocks);
                fw.resize(nblocks);
                is.resize(nblocks);
                xs.resize(nblocks);
                vs.resize(nblocks);
                fs.resize(nblocks);
            }

            firstStep = false;
        }

        // process flags
        adios2::Variable<char> in_vflags = 
            reader_io.InquireVariable<char>("flags");
        flags = in_vflags.Min();
        bool lfs = (flags % 2 == 1);
        flags = flags>>1;
        bool lvs = (flags % 2 == 1);
        flags = flags>>1;
        bool lxs = (flags % 2 == 1);
        flags = flags>>1;
        bool lfw = (flags % 2 == 1);
        flags = flags>>1;
        bool lvw = (flags % 2 == 1);
        flags = flags>>1;
        bool lxw = (flags % 2 == 1);
        flags = flags>>1;
        std::cout << "Rank " << rank << " flags :" 
            << lxw << lvw << lfw << lxs << lvs << lfs << std::endl;

        // Read trajectory data and indices
        in_viw = reader_io.InquireVariable<int64_t>("solvent/indices");
        if (lxw)
            in_vxw = reader_io.InquireVariable<double>("solvent/coords");
        if (lvw)
            in_vvw = reader_io.InquireVariable<double>("solvent/velocity");
        if (lfw)
            in_vfw = reader_io.InquireVariable<double>("solvent/forces");

        in_vis = reader_io.InquireVariable<int64_t>("solute/indices");
        if (lxs)
            in_vxs = reader_io.InquireVariable<double>("solute/coords");
        if (lvs)
            in_vvs = reader_io.InquireVariable<double>("solute/velocity");
        if (lfs)
            in_vfs = reader_io.InquireVariable<double>("solute/forces");

        for (size_t i = 0; i < nblocks; ++i)
        {

            std::cout << "Rank " << rank << " block " << i
                      << " select ID " << startBlockID+i << std::endl;
            in_viw.SetBlockSelection(startBlockID+i);
            reader.Get<int64_t>("solvent/indices", iw[i]);
            if (lxw) {
                in_vxw.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solvent/coords", xw[i]);
            }
            if (lvw) {
                in_vvw.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solvent/velocity", vw[i]);
            }
            if (lfw) {
                in_vfw.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solvent/forces", fw[i]);
            }

            in_vis.SetBlockSelection(startBlockID+i);
            reader.Get<int64_t>("solute/indices", is[i]);
            if (lxs) {
                in_vxs.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solute/coords", xs[i]);
            }
            if (lvs) {
                in_vvs.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solute/velocity", vs[i]);
            }
            if (lfs) {
                in_vfs.SetBlockSelection(startBlockID+i);
                reader.Get<double>("solute/forces", fs[i]);
            }
        }

        // Read rest of information
        reader.Get<int64_t>("solvent/nwmn", nwmn);
        reader.Get<int64_t>("solute/nsan", nsan);
        if (!rank) {
            if (firstStep) {
                reader.Get<std::string>("rdate", rdate);
            }
            reader.Get<std::string>("rtime", rtime);
            reader.Get<double>("stime", stime);
            reader.Get<double>("pres", pres);
            reader.Get<double>("temp", temp);
        }

        // End adios2 step for reading
        reader.EndStep();

        // NOTE: Input data is in memory at this point but not before 

        if (!rank) 
        {
            std::cout << "Sorting step " << stepSorting
                      << " processing NWCHEM output step " << stepSimOut
                      << " compute step " << stime << std::endl;
        }

        // Collect all data on rank 0 and sort there
        if (!rank)
        {
        }
        else
        {
            /* send
               int64_t nblocks, length of each block (iw[i].size()),
               iw[i], xw[i], vw[i], fw[i], is[i], xs[i], vs[i], fs[i]
            */
        }

        // Sort particles

        // Write out result

        if (!rank) {
            writer.BeginStep();
            writer.Put<double>(vstime, stime);
            writer.EndStep();
        }
        ++stepSorting;
    }


    // cleanup
    reader.Close();
    if (!rank) {
        writer.Close();
    }
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

    const unsigned int color = 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (argc < 2) {
        std::cout << "Not enough arguments\n";
        if (rank == 0) printUsage();
        MPI_Finalize();
        return 0;
    }
    std::string casename(argv[1]);

    work(casename);

    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}
