// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf. 
// 
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
// 
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org 
// or contact us at contact@libXerus.org.

/**
 * @file
 * @brief Implementation of some basic greedy contraction heuristics.
 */

#include <xerus/misc/check.h>

#include <xerus/contractionHeuristic.h>
#include <xerus/tensorNetwork.h>
#include <xerus/misc/internal.h>

namespace xerus {
    namespace internal {
		
		template<double (*scoreFct)(double, double, double, double, double)>
		void greedy_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network) {
			// estimated cost to calculate this heuristic is
			// numNodes * numNodes * 2*avgEdgesPerNode = 2 * numNodes * numEdges
			double numNodes = 0, numEdges = 0;
			for (const auto &node : _network.nodes) {
				if (!node.erased) {
					numNodes += 1;
					numEdges += static_cast<double>(node.degree());
				}
			}
			// if the best solution is only about twice as costly as the calculation of this heuristic, then don't bother
			if (_bestCost < 2 * 2 * numNodes * numEdges) { return; }
			
			double bestScore, ourCost=0;
			double ourFinalCost=0;
			std::vector<std::pair<size_t,size_t>> ourContractions;
			size_t bestId1, bestId2;
			do {
				bestScore = std::numeric_limits<double>::max();
				for (size_t i = 0; i < _network.nodes.size(); ++i) {
					if (_network.nodes[i].erased) { continue; }
					TensorNetwork::TensorNode &ni = _network.nodes[i];
					for (size_t j = i+1; j < _network.nodes.size(); ++j) {
						if (_network.nodes[j].erased) { continue; }
						TensorNetwork::TensorNode &nj = _network.nodes[j];
						/* possible candidate (i.e. link to a later node) */
						/* calculate n,m,r */
						double m=1,n=1,r=1;
						for (size_t d = 0; d < ni.degree(); ++d) {
							if (ni.neighbors[d].other == j) {
								r *= static_cast<double>(ni.neighbors[d].dimension);
							} else {
								m *= static_cast<double>(ni.neighbors[d].dimension);
							}
						}
						for (size_t d = 0; d < nj.degree(); ++d) {
							if (nj.neighbors[d].other != i) {
								n *= static_cast<double>(nj.neighbors[d].dimension);
							}
						}
						double tmpscore = scoreFct(m,n,r,0.0,0.0);
						if (tmpscore < bestScore) {
							bestScore = tmpscore;
							ourCost = contraction_cost(m,n,r,0.0,0.0);
							bestId1 = i;
							bestId2 = j;
						}
					}
				}
				if (bestScore < std::numeric_limits<double>::max()) {
					ourFinalCost += ourCost;
					if (ourFinalCost > _bestCost) {
						return;
					}
					ourContractions.emplace_back(bestId1,bestId2);
					_network.contract(bestId1,bestId2);
				}
			} while (bestScore < std::numeric_limits<double>::max());
			if (ourFinalCost < _bestCost) {
				_bestCost = ourFinalCost;
				_contractions = std::move(ourContractions);
			}
		}
		
		
		
		
		double contraction_cost(double _m, double _n, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			return _m*_n*_r; // TODO sparse
		}
		
		
		
		
		double score_size(double _m, double _n, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			return _n*_m-(_n+_m)*_r;
		}
		double score_mn(double _m, double _n, double  /*_r*/, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			return _m*_n;
		}
		double score_speed(double _m, double _n, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			return (_n*_m-(_n+_m)*_r)/(_n*_m*_r);
		}
		double score_r(double  /*_m*/, double  /*_n*/, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			return -_r;
		}
		double score_big_tensor(double _m, double _n, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			if (_n*_m<(_n+_m)*_r) {
				return -1e10 + _n*_m*_r;
			} 
				return _n*_m-(_n+_m)*_r;
			
		}
		double score_littlestep(double _m, double _n, double _r, double  /*_sparsity1*/, double  /*_sparsity2*/) {
			if (_n*_m<(_n+_m)*_r) {
				return -std::max(_n,_m)*_r;
			} 
				return _n*_m-(_n+_m)*_r;
			
		}
		
		
		
		std::tuple<size_t, size_t, size_t, double> best_of_three(const TensorNetwork &_network, size_t _id1, size_t _id2, size_t _id3) {
			const TensorNetwork::TensorNode &na = _network.nodes[_id1];
			const TensorNetwork::TensorNode &nb = _network.nodes[_id2];
			const TensorNetwork::TensorNode &nc = _network.nodes[_id3];
			double sa=1, sb=1, sc=1; // sizes devided by the link dimensions between a,b,c
			double sab=1, sbc=1, sac=1; // link dimensions
			for (size_t d = 0; d < na.degree(); ++d) {
				if (na.neighbors[d].links(_id2)) {
					sab *= static_cast<double>(na.neighbors[d].dimension);
				} else if (na.neighbors[d].links(_id3)) {
					sac *= static_cast<double>(na.neighbors[d].dimension);
				} else {
					sa *= static_cast<double>(na.neighbors[d].dimension);
				}
			}
			for (size_t d = 0; d < nb.degree(); ++d) {
				if (nb.neighbors[d].links(_id3)) {
					sbc *= static_cast<double>(nb.neighbors[d].dimension);
				} else if (!nb.neighbors[d].links(_id1)) {
					sb *= static_cast<double>(nb.neighbors[d].dimension);
				}
			}
			for (size_t d = 0; d < nc.degree(); ++d) {
//                 size_t other = nc.neighbors[d].other;
				if (!nc.neighbors[d].links(_id1) && !nc.neighbors[d].links(_id2)) {
					sc *= static_cast<double>(nc.neighbors[d].dimension);
				}
			}
			// cost of contraction a-b first etc.
			double costAB = sa*sb*sac*sbc*(sab+sc); // (sa*sac)*sab*(sb*sbc) + sa*sb*sac*sbc*sc;
			double costAC = sa*sc*sab*sbc*(sac+sb); 
			double costBC = sb*sc*sab*sac*(sbc+sa);
			if (costAB < costAC && costAB < costBC) {
				return std::tuple<size_t, size_t, size_t, double>(_id1, _id2, _id3, sa*sb*sac*sbc*sab);
			} 
			if (costAC < costBC) {
				return std::tuple<size_t, size_t, size_t, double>(_id1, _id3, _id2, sa*sc*sab*sbc*sac);
			}
			return std::tuple<size_t, size_t, size_t, double>(_id2, _id3, _id1, sb*sc*sab*sac*sbc);
		}
		
		
		void greedy_best_of_three_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network) {
			// estimated cost to calculate this heuristic is
			// numNodes * numNodes * 3*avgEdgesPerNode = 3 * numNodes * numEdges
			size_t numNodes=0, numEdges=0;
			for (const auto &node : _network.nodes) {
				if (!node.erased) {
					numNodes += 1;
					numEdges += node.degree();
				}
			}
			// if the best solution is only about twice as costly as the calculation of this heuristic, then don't bother
			if (_bestCost < double(2 * 3 * numNodes * numEdges)) { return; }
			
			double ourFinalCost=0;
			std::vector<std::pair<size_t,size_t>> ourContractions;
			while (numNodes >= 3) {
				// find a node with lowest degree
				size_t id1 = 0;
				size_t currDegree=~0ul;
				for (size_t i=0; i<_network.nodes.size(); ++i) {
					if (!_network.nodes[i].erased) {
						if (_network.nodes[i].degree() < currDegree) {
							id1 = i;
							currDegree = _network.nodes[i].degree();
						}
					}
				}
				
				// find its neighbor with lowest degree
				size_t id2 = 0;
				currDegree=~0ul;
				for (const TensorNetwork::Link &l : _network.nodes[id1].neighbors) {
					if (!l.external) {
						if (_network.nodes[l.other].degree() < currDegree) {
							id2 = l.other;
							currDegree = _network.nodes[l.other].degree();
						}
					}
				}
				
				size_t id3=0;
				// find the next available node
				while (id3 == id1 || id3 == id2 || _network.nodes[id3].erased) {
					id3 += 1;
				}
				size_t numConnections = 0;
				for (const TensorNetwork::Link &l : _network.nodes[id3].neighbors) {
					if (l.links(id1) || l.links(id2)) {
						numConnections += 1;
					}
				}
				// find the next most connected node
				for (size_t i=id3+1; i<_network.nodes.size(); ++i) {
					if (i == id1 || i == id2) { continue; }
					size_t newConnections=0;
					for (const TensorNetwork::Link &l : _network.nodes[i].neighbors) {
						if (l.links(id1) || l.links(id2)) {
							newConnections += 1;
						}
					}
					if (newConnections > numConnections) {
						numConnections = newConnections;
						id3 = i;
					}
				}
				
				// find the next best contraction within id1,id2,id3
				std::tuple<size_t, size_t, size_t, double> contraction = best_of_three(_network, id1, id2, id3);
				ourFinalCost += std::get<3>(contraction);
				if (ourFinalCost > _bestCost) {
					return;
				}
// 				if (std::get<1>(contraction) == id1) {
// 					id1 = id3;
// 				} else if (std::get<1>(contraction) == id2) {
// 					id2 = id3;
// 				}
				ourContractions.emplace_back(std::get<0>(contraction), std::get<1>(contraction));
				_network.contract(std::get<0>(contraction), std::get<1>(contraction));
				numNodes -= 1;
				LOG(cont, std::get<0>(contraction) << " " << std::get<1>(contraction));
				
				if (numNodes == 2) {
					ourFinalCost += _network.contraction_cost(id1, id2);
					ourContractions.emplace_back(id1, id2);
					LOG(cont, id1 << " " << id2);
					numNodes -= 1;
				}
			}
			
			LOG(hahaha, ourFinalCost << " vs " << _bestCost);
			if (ourFinalCost < _bestCost) {
				_bestCost = ourFinalCost;
				_contractions = std::move(ourContractions);
			}
		}
		
		
		void exchange_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network) {
			// estimated cost to calculate this heuristic is
			// numContractions * 3*avgEdgesPerNode ~= 3 * numEdges
			TensorNetwork copyNet(_network);
			double numEdges=0;
			for (const auto &node : _network.nodes) {
				if (!node.erased) {
					numEdges += static_cast<double>(node.degree());
				}
			}
			// if the best solution is only about twice as costly as the calculation of this heuristic, then don't bother
			double cost_of_heuristic = 3*numEdges;
// 			LOG(vergleich, _bestCost << " vs " << cost_of_heuristic);
			if (_bestCost < 2 * cost_of_heuristic) { return; }
			
			
			std::vector<std::pair<size_t, size_t>> openPairs;
			openPairs.push_back(_contractions.front());
			
			double ourFinalCost=0;
			std::vector<std::pair<size_t,size_t>> ourContractions;
			std::vector<size_t> idMap;
			for (size_t i =0; i<_network.nodes.size(); ++i) {
				idMap.emplace_back(i);
			}
			
			for (size_t i=1; i<_contractions.size(); ++i) {
				std::pair<size_t, size_t> next = _contractions[i];
				while (next.first != idMap[next.first]) { next.first = idMap[next.first]; }
				while (next.second != idMap[next.second]) { next.second = idMap[next.second]; }
				
				std::vector<std::pair<size_t, size_t>> newOpenPairs;
				for (const std::pair<size_t, size_t> &p : openPairs) {
					size_t id1 = p.first;
					while (id1 != idMap[id1]) { id1 = idMap[id1]; }
					size_t id2 = p.second;
					while (id2 != idMap[id2]) { id2 = idMap[id2]; }
					if (next.first != id1 && next.first != id2) {
						if (next.second == id1 || next.second == id2) {
							auto contr = best_of_three(_network, id1, id2, next.first);
							size_t a = std::get<0>(contr);
							size_t b = std::get<1>(contr);
							size_t c = std::get<2>(contr);
							idMap[b] = a;
							ourFinalCost += std::get<3>(contr);
							ourContractions.emplace_back(a,b);
							_network.contract(a,b);
							next.first = a;
							next.second = c;
						} else {
							newOpenPairs.emplace_back(id1,id2);
						}
					} else {
						if (next.second == id1 || next.second == id2) {
							LOG(fatal, "ie");
						} else {
							auto contr = best_of_three(_network, id1, id2, next.second);
							size_t a = std::get<0>(contr);
							size_t b = std::get<1>(contr);
							size_t c = std::get<2>(contr);
							idMap[b] = a;
							ourFinalCost += std::get<3>(contr);
							ourContractions.emplace_back(a,b);
							_network.contract(a,b);
							next.first = a;
							next.second = c;
						}
					}
				}
				newOpenPairs.emplace_back(next);
				openPairs = std::move(newOpenPairs);
			}
			
			INTERNAL_CHECK(openPairs.size() == 1, "ie");
			
			ourFinalCost += _network.contraction_cost(openPairs.front().first, openPairs.front().second);
			ourContractions.emplace_back(openPairs.front());
			
// 			LOG(hohohoEx, ourFinalCost << " vs " << _bestCost);
			if (ourFinalCost < _bestCost) {
				bool repeat = false;
				if (_bestCost - ourFinalCost > cost_of_heuristic * 2) {
					repeat = true;
				}
				_bestCost = ourFinalCost;
				_contractions = std::move(ourContractions);
				
				if (repeat) {
					exchange_heuristic(_bestCost, _contractions, copyNet);
				}
			}
		}
		
		
		const std::vector<ContractionHeuristic> contractionHeuristics {
			&greedy_heuristic<&score_size>,
			&greedy_heuristic<&score_mn>,
			&greedy_heuristic<&score_speed>,
// 			&greedy_heuristic<&score_r>,
			&greedy_heuristic<&score_big_tensor>,
			&greedy_heuristic<&score_littlestep>
// 			,&greedy_best_of_three_heuristic
			,&exchange_heuristic
		};
    } // namespace internal

} // namespace xerus
