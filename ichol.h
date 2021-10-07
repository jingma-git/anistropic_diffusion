#pragma once
#include <string>
#include <vector>
#include <memory>
#include <bitset>
#include <Eigen/Eigen>

// build chol_hierarchy
#include <boost/heap/binomial_heap.hpp>
#include "nanoflann.hpp"

// build sparse pattern

namespace ichol
{
    typedef double Float;
    typedef size_t Int;
    typedef Eigen::Matrix<Float, -1, -1> matd_t;
    typedef Eigen::Matrix<Float, -1, 1> vecd_t;
    typedef Eigen::Matrix<Int, -1, -1> mati_t;
    typedef Eigen::Matrix<Int, -1, 1> veci_t;
    typedef Eigen::SparseMatrix<Float> spmat_t;
    typedef Eigen::SparseMatrix<Float, Eigen::RowMajor> spmatr_t;
    typedef Eigen::PermutationMatrix<-1, -1> perm_t;
    typedef Eigen::SparseMatrix<char, Eigen::ColMajor, std::ptrdiff_t> patt_t; // for initial pattern, because it is usually very large
    typedef Eigen::SparseMatrix<std::ptrdiff_t> sn_patt_t;                     // supernode pattern type
    typedef Eigen::Matrix<Float, -1, -1, Eigen::RowMajor> matdr_t;             // row-major type, for kd-tree
    typedef Eigen::SparseMatrix<Float, Eigen::RowMajor> spmatr_t;
    typedef std::bitset<64> mask_t;

    namespace mschol
    {
        struct ichol_param
        {
            double rho = 7.0;
            double alpha = 0.0001;
        };

        struct chol_level
        {
            matd_t nods_; // reordered by a coarse to fine manner
            mati_t cell_; // only has meaning for the finest level

            // presribed refinement and its kernel to next level
            spmat_t C_;                   // #level_nods by #finer_level_nods
            spmat_t W_;                   // (#finer_level_nods-#level_nods) by #finer_level_nods
            const std::string mesh_type_; // trig/tet

            double calc_supp_scale(double vol);
        };

        // ordered in coarse-to-fine manner, last level has the most amount of nods
        class chol_hierarchy
        {
        public:
            // farthest-point sampling
            // heap, kd-tree(for pruning the close nods)
            typedef std::pair<Int, double> heap_data_t;
            struct heap_data_comp
            {
                bool operator()(const heap_data_t &lhs, const heap_data_t &rhs)
                {
                    return lhs.second < rhs.second;
                }
            };
            typedef boost::heap::binomial_heap<heap_data_t, boost::heap::compare<heap_data_comp>> heap_t;
            typedef heap_t::handle_type handle_t;
            typedef nanoflann::KDTreeEigenMatrixAdaptor<matdr_t> kd_tree_t;

            chol_hierarchy(const matd_t &nods, const mati_t &cell, const std::string &mesh_type);
            void build(std::vector<std::shared_ptr<chol_level>> &levels,
                       const Int coarsest_nods_num,
                       const Int prb_rd){};

        private:
            const matd_t &nods_;
            const mati_t &cell_;
            const std::string mesh_type_;

            std::shared_ptr<kd_tree_t> kdt_;
            matdr_t pts_;
        };

        // ordered in fine-to-coarse manner, the front cols of pattern matrix correspond to finest level
        class ichol_patt // pattern
        {
        public:
            ichol_patt(const std::vector<std::shared_ptr<chol_level>> &levels);
            void run(const double rho);

            const perm_t &getFullP() const { return fullP_; }
            const patt_t &getS0() const { return S0_; }
            const vecd_t &getL() const { return l_; }

        private:
            const std::vector<std::shared_ptr<chol_level>> &levels_;
            const matd_t &nods_;
            const Int Rd_; // 1 for trig_laplacian, 2 for tet_hession

            vecd_t l_;  // #nodes_ scale of nod_distance in each level
            patt_t S0_; // zeroS_ initial sparse pattern
            perm_t fullP_;
        };

        // reorder inital sparsity pattern to improve cache coherence
        struct geom_supernode
        {
            typedef std::ptrdiff_t index_t;

            perm_t aggregate(const std::vector<std::shared_ptr<chol_level>> &levels,
                             const vecd_t &l,
                             const perm_t &P,
                             const patt_t &S0,
                             const double rho,
                             const index_t su_size_bound);

            perm_t multicoloring();

            void calc_mask(const patt_t &S0,
                           const perm_t &P);

            std::vector<index_t> sn_level_ptr_; // how many supernodes on each level

            std::vector<index_t> sn_ptr_, sn_ind_; // how many nods in supernode, record 'S0 nod index' for nods in this supernode
            Eigen::SparseMatrix<index_t> S_su_;
            std::vector<mask_t> mask_;         // #nnz of S_su_
            Eigen::SparseMatrix<mask_t> S_MA_; // block mask for each supernode

        private:
            veci_t group_by_color(const Eigen::SparseMatrix<index_t> &A,
                                  const index_t lev_begin,
                                  const index_t lev_end);
        };

        // super nodal incomplete cholesky solver
        template <typename float_t, typename index_t, int ORDER = Eigen::ColMajor>
        struct supcol_sparse_matrix
        {
            std::vector<index_t> sup_ptr_, sup_ind_;
            std::vector<index_t> col2sup_, col2off_;
            Eigen::SparseMatrix<index_t, ORDER> sup_U_;
            Eigen::SparseMatrix<mask_t, ORDER> mask_;
            Eigen::Matrix<float_t, -1, 1> block_;
        };

        template <typename float_t, typename index_t, int ORDER = Eigen::ColMajor>
        struct supcol_sparse_matrix_ref
        {
            std::vector<index_t> *sup_ptr_, *sup_ind_;
            std::vector<index_t> *col2sup_, *col2off_;
            Eigen::SparseMatrix<index_t, ORDER> sup_U_;
            Eigen::SparseMatrix<mask_t, ORDER> mask_;
            Eigen::Matrix<float_t, -1, 1> *block_;
        };

        class supcol_ichol_solver
        {
        public:
            typedef Float float_t;
            typedef std::ptrdiff_t index_t;

            supcol_ichol_solver(const float_t ALPHA) : ALPHA_(ALPHA) {}
            void symbolic_phase(const Eigen::SparseMatrix<float_t> &A,
                                const Eigen::SparseMatrix<index_t> &supU,
                                const Eigen::SparseMatrix<mask_t> &mask,
                                const std::vector<index_t> &sn_ptr,
                                const std::vector<index_t> &sn_ind);

        public:
            const float_t ALPHA_;
            supcol_sparse_matrix<float_t, index_t, Eigen::ColMajor> SU_; // upper triangular
            supcol_sparse_matrix_ref<float_t, index_t, Eigen::RowMajor> SL_, SLT_;

            // record mapping between A and supernodal pattern
            // flatten system matrix A's nnz into continous array according to supernodal sparsity pattern
            std::vector<index_t> nnz_pos_;

        private:
            void init(const Eigen::SparseMatrix<index_t> &sn_U,
                      const Eigen::SparseMatrix<mask_t> &mask,
                      const std::vector<index_t> &sn_ptr,
                      const std::vector<index_t> &sn_ind,
                      supcol_sparse_matrix<float_t, index_t> &mat);

            index_t node_size(const supcol_sparse_matrix<float_t, index_t, Eigen::ColMajor> &mat, const index_t VID)
            {
                return mat.sup_ptr_[VID + 1] - mat.sup_ptr_[VID];
            }

            int sum_before(const mask_t &m, const int p);
        };

        class preconditioner
        {
        public:
            virtual ~preconditioner() {}
            virtual int analyse_pattern(const matd_t &mat) = 0;
            virtual int factorize(const matd_t &mat, const bool verbose = true) = 0;
            int compute(const matd_t &mat, const bool verbose = true)
            {
                int rtn = 0;
                rtn |= analyse_pattern(mat);
                rtn |= factorize(mat, verbose);
                return rtn;
            }
            virtual vecd_t solve(const vecd_t &rhs) = 0;
        };

        class ichol_precond : preconditioner
        {
        public:
            typedef supcol_ichol_solver solver_t;

            ichol_precond(const std::vector<std::shared_ptr<chol_level>> &levels,
                          const ichol_param &pt);

            int analyse_pattern(const spmat_t &mat);
            int factorize(const spmat_t &mat, const bool verbose);

        private:
            const ichol_param &pt_;
            // levels
            const std::vector<std::shared_ptr<chol_level>> &levels_;
            std::vector<std::ptrdiff_t> level_ptr_; // fine to coarse

            // pattern builder: geom_supernode, permutation matrix
            std::shared_ptr<geom_supernode> gs_;
            perm_t P_;

            // solver
            std::shared_ptr<solver_t> ic_slv_;
        };

        class precond_cg_solver
        {
        public:
            typedef Eigen::Index Index; // std::ptrdiff_t

            precond_cg_solver()
            {
                tol_ = 1e-12;
            }

            precond_cg_solver(const std::shared_ptr<preconditioner> &precond) : precond_(precond)
            {
                tol_ = 1e-12;
            }

            int analyse_pattern(const spmat_t &mat)
            {
                return precond_->analyse_pattern(mat);
            }
            int factorize(const spmat_t &mat, const bool verbose = true)
            {
                mat_ = &mat;
                maxits_ = 2 * mat.cols();
                err_.reserve(maxits_);
                time_.reserve(maxits_);
                return precond_->factorize(mat, verbose);
            }

            void set_maxits(const Index maxits)
            {
                maxits_ = maxits;
                err_.reserve(maxits_);
                time_.reserve(maxits_);
            }

            void set_tol(const float_t tol)
            {
                tol_ = tol;
            }

        protected:
            //-> final error and real iteration number
            float_t m_tol_error;
            Index m_iters;

            spmat_t const *mat_;
            const std::shared_ptr<preconditioner> precond_;
            std::vector<float_t> err_, time_;

            Index maxits_;
            float_t tol_;
        };
    }

    namespace problem
    {
        class smooth2d // edge-preserved smoothing (image)
        {
        };

        class smooth3d // smooth a mesh (? how about edge-preserved smoothing of mesh)
        {
        public:
            smooth3d(const matd_t &nods, const mati_t &cell)
            {
                // compute system matrix
            }

        private:
            spmat_t A_; // laplace matrix
        };
    }
}