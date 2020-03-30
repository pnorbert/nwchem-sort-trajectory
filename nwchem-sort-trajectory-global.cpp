/*
 * Trajectory sorting code for NWCHEM unsorted ADIOS2 trajectory data
 * Reads <CASENAME>_trj_dump.bp and writes <CASENAME>_trj.bp
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

size_t sum_sizes(std::vector<int64_t> &array)
{
    return static_cast<size_t>(std::accumulate(array.begin(), array.end(), 0));
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

/* Serialize all data blocks into a table
 * This means transposing all elements so that one atom/molecule
 * is in one row.
 * nmolecules = # of solvent molecules or solute atoms
 * nvalues    = # of coordinates per value, 3 for data, 1 for indices
 * nwa        = # of atoms in solvent molecules, 1 for solute
 */
template <class T>
std::vector<T> make_table(bool flag, adios2::Variable<T> &v,
                          std::vector<std::vector<T>> &arrays, const int nelems,
                          const int nvalues, const int nwa)
{
    std::vector<T> table;
    if (flag)
    {
        const size_t recordsize = nvalues * nwa;
        table.resize(nelems * recordsize);
        size_t currentIdx = 0;
        for (int i = 0; i < arrays.size(); ++i)
        {
            std::vector<T> &a = arrays[i];
            const size_t n = a.size() / nvalues / nwa;
            // pre-calculate the offsets from where we copy pieces of info
            // for one molecule/atom
            size_t ns[recordsize];
            ns[0] = 0;
            for (int j = 1; j < recordsize; ++j)
            {
                ns[j] = ns[j - 1] + n;
            }

            std::cout << "-- Rank " << rank << " copy " << n
                      << " elements to var " << v.Name()
                      << " table nrows = " << nelems
                      << " recordsize = " << recordsize
                      << " at index = " << currentIdx << " ns = [";
            for (int j = 0; j < recordsize; ++j)
            {
                std::cout << ns[j] << " ";
            }
            std::cout << "]" << std::endl;
            /*std::cout << "-- Rank " << rank << " arrays[" << i << "] = [";
            for (int j = 0; j < arrays[i].size(); ++j)
            {
                std::cout << " " << arrays[i][j];
            }
            std::cout << "] " << std::endl;*/

            for (size_t j = 0; j < n; ++j)
            {
                for (size_t k = 0; k < nwa; ++k)
                {
                    for (size_t l = 0; l < nvalues; ++l)
                    {
                        table[currentIdx] = a[j + ns[k * nvalues + l]];
                        ++currentIdx;
                    }
                }
            }
        }
        if (currentIdx != table.size())
        {
            std::cout << " ERROR: currentIDX=" << currentIdx
                      << " should be equal to "
                      << " table size = " << table.size() << std::endl;
        }
    }
    return table;
}

bool epsilon(double d) { return (fabs(d) < 1.0e-20); }
bool epsilon(int64_t d) { return (d == 0); }

/* Gather one array on 'root' process  */
template <class T>
void dbgCheckZeros(bool flag, adios2::Variable<T> &v,
                   std::vector<T> &mydata, std::vector<int64_t> &myindex,
                   int recordsize, int root)
{
    int firstZeroPos=-1;
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

/* Gather one array on 'root' process  */
std::pair<std::vector<double>, std::vector<int64_t>>
gather_array(bool flag, adios2::Variable<double> &v,
             std::vector<double> &mydata, std::vector<int64_t> &myindex,
             int recordsize, int root)
{
    std::vector<double> alldata;
    std::vector<int64_t> allindex;
    const size_t nMyElems = myindex.size();

    if (flag)
    {
        // variables for MPI Gather and Gatherv
        int iElems = static_cast<int>(nMyElems);
        int recvCountData[comm_size], recvCountIdx[comm_size];
        int displData[comm_size], displIdx[comm_size];

        // Gather sizes of all blocks from all processes
        MPI_Gather(&iElems, 1, MPI_INT, &recvCountIdx, 1, MPI_INT, root, comm);
        if (rank == root)
        {
            // MPI displacement calculation
            int nTotal = 0;
            for (int i = 0; i < comm_size; ++i)
            {
                displIdx[i] = nTotal;
                displData[i] = nTotal * recordsize;
                recvCountData[i] = recvCountIdx[i] * recordsize;
                nTotal += recvCountIdx[i];
            }

            // Gather all blocks from all processes
            alldata.resize(static_cast<size_t>(nTotal * recordsize));
            allindex.resize(static_cast<size_t>(nTotal));

            std::cout << "-- Rank " << rank << " gathers data and indices of "
                      << nTotal << " elements, recordsize = " << recordsize
                      << std::endl;
        }

        std::cout << "-- Rank " << rank << " receiveCountIdx = [";
        for (int i = 0; i < comm_size; ++i)
        {
            std::cout << " " << recvCountIdx[i];
        }
        std::cout << "] " << std::endl;

        std::cout << "-- Rank " << rank << " recvCountData = [";
        for (int i = 0; i < comm_size; ++i)
        {
            std::cout << " " << recvCountData[i];
        }
        std::cout << "] " << std::endl;

        MPI_Gatherv(mydata.data(), iElems * recordsize, MPI_DOUBLE,
                    alldata.data(), recvCountData, displData, MPI_DOUBLE, root,
                    comm);
        MPI_Gatherv(myindex.data(), iElems, MPI_INT64_T, allindex.data(),
                    recvCountIdx, displIdx, MPI_INT64_T, root, comm);
    }
    return std::make_pair(alldata, allindex);
}

/* Sort and write on root process
 * Here we have a table where each row is one molecule/atom
 */
void sort_and_write_array(
    bool flag, adios2::Variable<double> &v, adios2::Engine &writer,
    std::pair<std::vector<double>, std::vector<int64_t>> serializedData,
    int recordsize, int root)
{
    if (flag && rank == root)
    {
        std::vector<double> array = serializedData.first;
        std::vector<int64_t> indices = serializedData.second;
        std::vector<double> sorted;

        const size_t n = indices.size();

        sorted.resize(n * recordsize);
        assert(sorted.size() == array.size());

        std::cout << "Rank " << rank << " sort " << v.Name() << " " << n
                  << " elements" << std::endl;

#if 1
        // Note: indices run 1..N fortran style, here we calculate with 0..N-1
        for (size_t i = 0; i < n; ++i)
        {
            size_t idx = (indices[i] - 1);
            if (idx >= n)
            {
                std::cout << "-- Rank " << rank << " indices[" << i
                          << "] = " << indices[i] - 1 << " idx = " << idx
                          << " is out of range = " << n << std::endl;
            }
            assert(idx < n);
            std::copy(array.begin() + i * recordsize,
                      array.begin() + (i + 1) * recordsize,
                      sorted.begin() + idx * recordsize);
        }
#else
        /* Copy as is. No sorting */
        for (size_t i = 0; i < n; ++i)
        {
            size_t src = i * recordsize;
            for (size_t j = 0; j < recordsize; ++j)
            {
                // std::cout << "-- Rank " << rank << " sorted[ " << src
                //          << "] = array[src] = " << array[src] << std::endl;
                sorted[src] = array[src++];
            }
        }
#endif
        writer.Put(v, sorted.data(), adios2::Mode::Sync);
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
    int64_t nsa;

    // Number of Solvent molecules per process including garbage data
    // i.e. size of solvent data blocks on each process
    int64_t mwm;
    // Number of Solute atoms per process including garbage data
    // i.e. size of solute data blocks on each process
    int64_t msa;

    /* Data changing every step */

    // Actual number of solvent molecules per process, changing per step
    std::vector<int64_t> nwmn;
    // Actual number of solute atoms per process, changing per step
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
    std::vector<double> xw, vw, fw;
    // solute data is 2D (3 x nsan)
    std::vector<double> xs, vs, fs;
    // indices (id of solvent molecules and solute atoms)
    std::vector<int64_t> iw, is;

    // flags to indicate presence of a quantity
    bool lfs, lvs, lxs, lfw, lvw, lxw;

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
    size_t nblocks, startBlockID;
    // sizes and offsets in array to read in
    size_t startw, countw;
    size_t starts, counts;

    bool firstStep = true;

    // adios2 variable declarations for some input variables
    adios2::Variable<int64_t> in_vnwmn, in_vnsan;
    adios2::Variable<double> in_vxw, in_vvw, in_vfw, in_vxs, in_vvs, in_vfs;
    adios2::Variable<int64_t> in_viw, in_vis;

    // adios2 variable declarations for the output variables
    adios2::Variable<int64_t> vnwm, vnwa, vnsa, vmwm, vmsa;
    adios2::Variable<double> vxw, vvw, vfw, vxs, vvs, vfs;
    adios2::Variable<std::string> vrdate, vrtime;
    adios2::Variable<double> vstime, vpres, vtemp, vvlat;

    // adios2 io object and engine init
    adios2::ADIOS ad("adios2.xml", comm, adios2::DebugON);

    // IO objects for reading and writing
    adios2::IO reader_io = ad.DeclareIO("trj");
    // We use the IO and Variable definitions on all processes but only
    // rank 0 will use it for writing output
    adios2::IO writer_io = ad.DeclareIO("SortingOutput");
    if (!rank)
    {
        std::cout << "Sorter reads " << in_filename
                  << " using engine type: " << reader_io.EngineType()
                  << std::endl;
        std::cout << "Sorter writes " << out_filename
                  << " using engine type:      " << writer_io.EngineType()
                  << std::endl;
    }

    // Engines for reading and writing
    adios2::Engine reader =
        reader_io.Open(in_filename, adios2::Mode::Read, comm);
    adios2::Engine writer;
    if (!rank)
    {
        writer =
            writer_io.Open(out_filename, adios2::Mode::Write, MPI_COMM_SELF);
    }

    // read data per timestep
    int stepSorting = 0;
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
            adios2::Variable<int64_t> in_vmwm =
                reader_io.InquireVariable<int64_t>("solvent/mwm");
            adios2::Variable<int64_t> in_vmsa =
                reader_io.InquireVariable<int64_t>("solute/msa");


            nwriters = in_vnwriters.Min();
            nwm = in_vnwm.Min();
            nwa = in_vnwa.Min();
            nsa = in_vnsa.Min();
            mwm = in_vmwm.Min();
            msa = in_vmsa.Min();

            // Declare variables to output
            // We declare them on every process but only write them on one
            size_t snwm = static_cast<size_t>(nwm);
            size_t snwa = static_cast<size_t>(nwa);
            vxw = writer_io.DefineVariable<double>("solvent/coords",
                                                   {snwm, snwa, 3}, {0, 0, 0},
                                                   {snwm, snwa, 3}, false);
            vvw = writer_io.DefineVariable<double>("solvent/velocity",
                                                   {snwm, snwa, 3}, {0, 0, 0},
                                                   {snwm, snwa, 3});
            vfw = writer_io.DefineVariable<double>(
                "solvent/forces", {snwm, snwa, 3}, {0, 0, 0}, {snwm, snwa, 3});

            size_t snsa = static_cast<size_t>(nsa);
            vxs = writer_io.DefineVariable<double>("solute/coords", {snsa, 3},
                                                   {0, 0}, {snsa, 3});
            vvs = writer_io.DefineVariable<double>("solute/velocity", {snsa, 3},
                                                   {0, 0}, {snsa, 3});
            vfs = writer_io.DefineVariable<double>("solute/forces", {snsa, 3},
                                                   {0, 0}, {snsa, 3});

            if (!rank)
            {
                vnwm = writer_io.DefineVariable<int64_t>("solvent/nwm");
                vnwa = writer_io.DefineVariable<int64_t>("solvent/nwa");
                vnsa = writer_io.DefineVariable<int64_t>("solute/nsa");

                vrdate = writer_io.DefineVariable<std::string>("rdate");
                vrtime = writer_io.DefineVariable<std::string>("rtime");

                vstime = writer_io.DefineVariable<double>("stime");
                vpres = writer_io.DefineVariable<double>("pres");
                vtemp = writer_io.DefineVariable<double>("temp");

                vvlat = writer_io.DefineVariable<double>("vlat", {3, 3}, {0, 0},
                                                         {3, 3});
            }


            // How many blocks to read on this process?
            nblocks = nwriters/ comm_size;
            size_t extras =
            		static_cast<size_t>(nwriters % comm_size);
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

            // Pre-calculate offsets and sizes to read from global arrays
            startw = startBlockID * mwm;
            countw = nblocks * mwm;
            starts = startBlockID * msa;
            counts = nblocks * msa;


            std::cout << "Rank " << rank << " reads " << nblocks
            		<< "  blocks from idx " << startBlockID
					<< " from total " << nwriters
					<< " Reads " << countw
					<< "  solvent elements from offset " << startw
					<< " and reads " << counts << "solute elements from offset " << starts
					<< std::endl;

            iw.resize(nblocks*mwm);
            xw.resize(nwa*3*nblocks*mwm);
            vw.resize(nwa*3*nblocks*mwm);
            fw.resize(nwa*3*nblocks*mwm);
            is.resize(nblocks*msa);
            xs.resize(3*nblocks*msa);
            vs.resize(3*nblocks*msa);
            fs.resize(3*nblocks*msa);
            nwmn.resize(nblocks);
            nsan.resize(nblocks);


            // process flags
            adios2::Variable<int8_t> in_vflags =
            		reader_io.InquireVariable<int8_t>("flags");
            flags = static_cast<char>(in_vflags.Min());
            lfs = (flags % 2 == 1);
            flags = flags >> 1;
            lvs = (flags % 2 == 1);
            flags = flags >> 1;
            lxs = (flags % 2 == 1);
            flags = flags >> 1;
            lfw = (flags % 2 == 1);
            flags = flags >> 1;
            lvw = (flags % 2 == 1);
            flags = flags >> 1;
            lxw = (flags % 2 == 1);
            flags = flags >> 1;
            std::cout << "Rank " << rank << " flags :" << lxw << lvw << lfw << lxs
            		<< lvs << lfs << std::endl;

        }


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

        size_t snwa = static_cast<size_t>(nwa);

        /* Read solvent data, 'nblocks' complete consecutive blocks (from nblocks writers) */
        std::cout << "Rank " << rank << " reads solvent blocks " << startBlockID
        		<< ".." << startBlockID+nblocks-1 << std::endl;
        in_viw.SetSelection({{startw},{countw}});
        reader.Get<int64_t>(in_viw, iw);
        std::cout << "Rank " << rank
        		<< " reads solvent/indices"
				<< " shape = " << printDims(in_viw.Shape())
				<< " dims = " << printDims(in_viw.Count())
				<< " offset = " << printDims(in_viw.Start())
				<< std::endl;
        if (lxw)
        {
        	in_vxw.SetSelection({{0,0,startw},{snwa,3,countw}});
        	reader.Get<double>("solvent/coords", xw);
        	std::cout << "Rank " << rank
        			<< " reads solvent/coords "
					<< " shape = " << printDims(in_vxw.Shape())
					<< " dims = " << printDims(in_vxw.Count())
					<< " offset = " << printDims(in_vxw.Start())
					<< std::endl;
        }
        if (lvw)
        {
        	in_vvw.SetSelection({{0,0,startw},{snwa,3,countw}});
        	reader.Get<double>(in_vvw, vw);
        }
        if (lfw)
        {
        	in_vfw.SetSelection({{0,0,startw},{snwa,3,countw}});
        	reader.Get<double>(in_vfw, fw);
        }

        std::cout << "Rank " << rank << " reads solute blocks " << startBlockID
        		<< ".." << startBlockID+nblocks-1 << std::endl;
        in_vis.SetSelection({{starts},{counts}});
        reader.Get<int64_t>(in_vis, is);
        if (lxs)
        {
        	in_vxs.SetSelection({{0,0,starts},{3,counts}});
        	reader.Get<double>(in_vxs, xs);
        }
        if (lvs)
        {
        	in_vvs.SetSelection({{0,0,starts},{3,counts}});
        	reader.Get<double>(in_vvs, vs);
        }
        if (lfs)
        {
        	in_vfs.SetSelection({{0,0,starts},{3,counts}});
        	reader.Get<double>(in_vfs, fs);
        }


        // Read the actual per-process sizes
        in_vnwmn = reader_io.InquireVariable<int64_t>("solvent/nwmn");
        in_vnwmn.SetSelection({{startBlockID},{nblocks}});
        reader.Get<int64_t>(in_vnwmn, nwmn);

        in_vnsan = reader_io.InquireVariable<int64_t>("solute/nsan");
        in_vnsan.SetSelection({{startBlockID},{nblocks}});
        reader.Get<int64_t>(in_vnsan, nsan);

        // Read rest of information
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

        // NOTE: Input data is in memory at this point but not before
        /*for (size_t i = 0; i < nblocks_solvent; ++i)
        {
            dbgCheckZeros(true, in_viw, iw[i], iw[i], 1, rank);
            if (lxw)
            {
                assert(xw[i].size() == iw[i].size() * 3 * nwa);
                dbgCheckZeros(lxw, vxw, xw[i], iw[i], 3 * nwa, rank);
            }
            if (lvw)
                assert(vw[i].size() == iw[i].size() * 3 * nwa);
            if (lfw)
                assert(fw[i].size() == iw[i].size() * 3 * nwa);
        }
        for (size_t i = 0; i < nblocks_solute; ++i)
        {
            if (lxs)
                assert(xs[i].size() == is[i].size() * 3);
            if (lvs)
                assert(vs[i].size() == is[i].size() * 3);
            if (lfs)
                assert(fs[i].size() == is[i].size() * 3);
        }*/
        // std::cout << "Rank " << rank << " Done assert " << std::endl;

        if (!rank)
        {
            std::cout << "Sorting step " << stepSorting
                      << " processing NWCHEM output step " << stepSimOut
                      << " compute step " << stime << std::endl;
        }

        // Serialize blocks into one 2D array where
        // all data per particle is in one record ('t' is for 'table')
        size_t nSolventMoleculesLocal = sum_sizes(nwmn);

        std::vector<int64_t> tiw =
            make_table(true, in_viw, iw, nSolventMoleculesLocal, 1, 1);
        std::vector<double> txw =
            make_table(lxw, vxw, xw, nSolventMoleculesLocal, 3, nwa);
        std::vector<double> tvw =
            make_table(lvw, vvw, vw, nSolventMoleculesLocal, 3, nwa);
        std::vector<double> tfw =
            make_table(lfw, vfw, fw, nSolventMoleculesLocal, 3, nwa);

        size_t nSoluteAtomsLocal = sum_sizes(nsan);

        std::vector<int64_t> tis =
            make_table(true, in_vis, is, nSoluteAtomsLocal, 1, 1);
        std::vector<double> txs =
            make_table(lxs, vxs, xs, nSoluteAtomsLocal, 3, 1);
        std::vector<double> tvs =
            make_table(lvs, vvs, vs, nSoluteAtomsLocal, 3, 1);
        std::vector<double> tfs =
            make_table(lfs, vfs, fs, nSoluteAtomsLocal, 3, 1);

        dbgCheckZeros(lxw, vxw, txw, tiw, 3 * nwa, rank);
        dbgCheckZeros(lxs, vxs, txs, tis, 3, rank);

        // Collect each array on various ranks and sort there
        std::pair<std::vector<double>, std::vector<int64_t>> gxwi =
            gather_array(lxw, vxw, txw, tiw, nwa * 3, 0);
        std::pair<std::vector<double>, std::vector<int64_t>> gvwi =
            gather_array(lvw, vvw, tvw, tiw, nwa * 3, 0);
        std::pair<std::vector<double>, std::vector<int64_t>> gfwi =
            gather_array(lfw, vfw, tfw, tiw, nwa * 3, 0);
        std::pair<std::vector<double>, std::vector<int64_t>> gxsi =
            gather_array(lxs, vxs, txs, tis, 3, 0);
        std::pair<std::vector<double>, std::vector<int64_t>> gvsi =
            gather_array(lvs, vvs, tvs, tis, 3, 0);
        std::pair<std::vector<double>, std::vector<int64_t>> gfsi =
            gather_array(lfs, vfs, tfs, tis, 3, 0);

        // Sort particles
        // Write out result

        if (!rank)
        {
            writer.BeginStep();
        }

        sort_and_write_array(lxw, vxw, writer, gxwi, nwa * 3, 0);
        sort_and_write_array(lvw, vvw, writer, gvwi, nwa * 3, 0);
        sort_and_write_array(lfw, vfw, writer, gfwi, nwa * 3, 0);
        sort_and_write_array(lxs, vxs, writer, gxsi, 3, 0);
        sort_and_write_array(lvs, vvs, writer, gvsi, 3, 0);
        sort_and_write_array(lfs, vfs, writer, gfsi, 3, 0);

        if (!rank)
        {
            if (firstStep)
            {
                writer.Put<int64_t>(vnwm, nwm);
                writer.Put<int64_t>(vnwa, nwa);
                writer.Put<int64_t>(vnsa, nsa);
                writer.Put<std::string>(vrdate, rdate);
            }
            writer.Put<double>(vstime, stime);
            writer.Put<double>(vpres, pres);
            writer.Put<double>(vtemp, temp);
            writer.Put<std::string>(vrtime, rtime);
            writer.Put<double>(vvlat, vlat.data());
            writer.EndStep();
        }
        if (!stepSorting)
        {
        	reader.LockReaderSelections();
        }
        ++stepSorting;
        firstStep = false;
    }

    // cleanup
    reader.Close();
    if (!rank)
    {
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
