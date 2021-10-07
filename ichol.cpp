#include "ichol.h"
using namespace std;

namespace ichol
{
    namespace mschol
    {
        chol_hierarchy::chol_hierarchy(const matd_t &nods, const mati_t &cell, const std::string &mesh_type)
            : nods_(nods), cell_(cell), mesh_type_(mesh_type)
        {
            //-> build kd-tree
            pts_.resize(nods.cols(), nods.rows()); // #nods by #vertex_dim
            std::copy(nods.data(), nods.data() + nods.size(), pts_.data());
            kdt_ = std::make_shared<kd_tree_t>(pts_.cols(), pts_, 10); // #vertex_dim, data, #neighbors to query
            kdt_->index->buildIndex();
        }

        ichol_patt::ichol_patt(const std::vector<std::shared_ptr<chol_level>> &levels)
            : levels_(levels),
              nods_(levels.back()->nods_),
              Rd_(levels.back()->C_.cols() / levels.back()->nods_.cols())
        {
            // l_: calc length scale for each level
            // finest length scale
            // starting from finest level, and update with max-length-scale on that level
        }

        void ichol_patt::run(const double rho)
        {
            // fullP_:
            // level_idx
            // adj_list: range_search from finest level to coarsest level dij<rho*min(l[i], l[j])
            //           maps last_node of finest level to 1st_col_of_patt_matrix

            // build initial pattern according to adj_list
            vector<ptrdiff_t> ptr, ind; // #prb_dim * # nods + 1, how many cols in system matrix
            vector<char> val;

            // set S0 according to outer_index_ptr, inner_indices, and vals
        }

        perm_t geom_supernode::aggregate(const std::vector<std::shared_ptr<chol_level>> &levels,
                                         const vecd_t &l,
                                         const perm_t &P,
                                         const patt_t &S0,
                                         const double rho,
                                         const index_t su_size_bound)
        {
            const matd_t &nods = levels.back()->nods_;
            const int dim = S0.rows();
            const int Rd = S0.rows() / nods.cols();
            // symmetrize S0
            std::vector<std::vector<std::pair<index_t, index_t>>> S_col_iter; // row_i's adj_j and (i, adj_j)'s outerindex

            // group cols to supernodes: dist(i,j) < rho * min(l[i], l[j]), group i,j to the same supernode
            // input: S0, S_col_iter, output: su_ptr_, su_ind_, su_level_ptr_
            // for each level
            //     for nod_k in lev_begin:lev_end
            //        if nod_k is grouped: continue; else: start a new supernode (su_ptr_.push_back(su_ptr_.back()))
            //        search nod_k's row: if dist(k,j) < rho * min(l[k], l[j]) group nodj to this supernode (su_ptr_.back()++, su_ind_.push_back(j))
            //        search nod_k's col: if dist(k,i) < rho * min(l[k], l[i]) group nodi to this supernode (su_ptr_.back()++, su_ind_.push_back(i))

            // reorder to group supernodes into contiguous block
            perm_t P_su(dim);
        }

        perm_t geom_supernode::multicoloring()
        {
            // for each level, the supernodes are aggregated into multiple clusters according to preassigned colors,
            // as long as two adjacent nodes have different colors
            // each cluster could be processed parrallelly since no nodes in this cluster are depdent to each other
            Eigen::SparseMatrix<index_t> sym_su = S_su_.selfadjointView<Eigen::Upper>();

            // output: new order ajdusted by multicoloring algorithm
            // for lev in sn_levels (sn_level_ptr_)
            //     group_by_color(sym_su, lev_begin, lev_end)

            // sn_size, perm_sn_size, perm_sn_ptr, perm_sn_ind
            perm_t fullcolorP(sn_ind_.size());
            // set sn_ptr_, sn_ind_ as perm_sn_ptr, perm_sn_ind
            return fullcolorP;
        }

        void geom_supernode::calc_mask(const patt_t &S0,
                                       const perm_t &P)
        {
            // col_to_su, col_to_of(index in each supernodal block)

            // map supernode (I,J) to I * su_num + J

            // mask: S0, P
            vector<mask_t> tmp_mask(S_su_.nonZeros());
            S_MA_ = Eigen::Map<Eigen::SparseMatrix<mask_t>>(S_su_.rows(), S_su_.cols(),
                                                            S_su_.nonZeros(),
                                                            S_su_.outerIndexPtr(),
                                                            S_su_.innerIndexPtr(),
                                                            &tmp_mask[0]);
        }

        void supcol_ichol_solver::symbolic_phase(const Eigen::SparseMatrix<float_t> &A,
                                                 const Eigen::SparseMatrix<index_t> &supU,
                                                 const Eigen::SparseMatrix<mask_t> &mask,
                                                 const std::vector<index_t> &sn_ptr,
                                                 const std::vector<index_t> &sn_ind)
        {
            // init: copy sn_ptr, sn_ind, mask, supU, to SU_, fill SU_.block by counting elements of all supernode
            // multi_thread_level_scheduling, build direct_acyclic_graph, forward and backward
            // map A's nnz to nnz_pos_ assisted by SU_, nnz_pos_.size()==A.nonZeros()

            nnz_pos_.resize(A.nonZeros());
            std::fill(nnz_pos_.begin(), nnz_pos_.end(), -1);
            // #pragma omp parallel for
            for (index_t j = 0; j < A.cols(); ++j)
            {
                index_t iter = A.outerIndexPtr()[j];
                index_t J = SU_.col2sup_[j]; // supernode
                index_t ITER = SU_.sup_U_.outerIndexPtr()[J];
                while (iter < A.outerIndexPtr()[j + 1] && ITER < SU_.sup_U_.outerIndexPtr()[J + 1])
                {
                    index_t i = A.innerIndexPtr()[iter];
                    index_t I = SU_.sup_U_.innerIndexPtr()[ITER];
                    if (SU_.col2sup_[i] == I)
                    {
                        index_t numI = node_size(SU_, I);
                        index_t off_i = SU_.col2off_[i];
                        index_t off_j = SU_.col2off_[j];
                        index_t real_off_j = sum_before(SU_.mask_.valuePtr()[ITER], off_j);
                        nnz_pos_[iter] = SU_.sup_U_.valuePtr()[ITER] + off_i + real_off_j * numI;
                        ++iter;
                    }
                    else if (SU_.col2sup_[i] < I)
                    {
                        ++iter;
                    }
                    else
                    {
                        ++ITER;
                    }
                }
            }
        }

        void supcol_ichol_solver::init(const Eigen::SparseMatrix<index_t> &sn_U,
                                       const Eigen::SparseMatrix<mask_t> &mask,
                                       const std::vector<index_t> &sn_ptr,
                                       const std::vector<index_t> &sn_ind,
                                       supcol_sparse_matrix<float_t, index_t> &mat)
        {
            // fill mat.block (make supernode in a continous memory block)
        }

        int supcol_ichol_solver::sum_before(const mask_t &m, const int p)
        {
            int cnt = 0;
            for (int i = 0; i < p; ++i)
            {
                if (m[i])
                {
                    ++cnt;
                }
            }
            return cnt;
        }

        ichol_precond::ichol_precond(const std::vector<std::shared_ptr<chol_level>> &levels,
                                     const ichol_param &pt)
            : levels_(levels), pt_(pt)
        {
            //==================== build pattern =============================================
            // level_ptr_: fine to coarse
            // 1. multi-level ordering : ichol_patt, fine-to-coarse ordering
            std::shared_ptr<ichol_patt> order_patt = std::make_shared<ichol_patt>(levels);
            order_patt->run(pt.rho);
            P_ = order_patt->getFullP();

            // 2. reorder to improve cache-coherence: geom_supernode
            //    2a. aggregation ordering: supernodal
            //    2b. multi-coloring: increase supernodal independence
            //    2c. supernodal mask

            // ==================== build solver =============================================
            ic_slv_ = std::make_shared<supcol_ichol_solver>(pt.alpha);
        }

        int ichol_precond::analyse_pattern(const spmat_t &mat)
        {
            Eigen::SparseMatrix<Float> symbA;
            symbA = mat.twistedBy(P_);

            ic_slv_->symbolic_phase(symbA, gs_->S_su_, gs_->S_MA_, gs_->sn_ptr_, gs_->sn_ind_);
        }

        int ichol_precond::factorize(const spmat_t &mat, const bool verbose)
        {
        }
    }
}
