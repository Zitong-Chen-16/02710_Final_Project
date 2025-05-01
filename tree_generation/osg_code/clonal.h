//
//  clonal.h
//  Clonal_interference
//
//  Created by Yang Ping Kuo on 9/9/22.
//

#pragma once
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include<cmath>

#include <random>
#include <vector>
#include <unordered_map>
#include<string>
#include <fstream>
#include <iostream>

using namespace std;

// Graph data structure
class Structure
{
public:
    int popsize;
    int *degrees, **edgelist;
    Structure(string);
    //~Structure();
};
// Constructor: takes txt file containing edgelist of a graph
Structure::Structure(string input_name)
{
    ifstream input(input_name);

    vector<int> out, in;
    int node;
    int i = 0;
    popsize = 1000;
    while (input >> node)
    {
        // popsize = popsize < node ? node: popsize;
        if (i % 2 == 0)
            out.push_back(node);
        else
            in.push_back(node);
        ++i;
    }
    if (out.size() != in.size())
        throw invalid_argument("In and out should have same length");
    // ++popsize;
    
    degrees = new int[popsize];
    int *temp = new int[popsize];
    
    for (int node = 0; node < popsize; ++node)
    {
        temp[node] = 0;
        degrees[node] = 0;
    }
    
    for (auto node : out)
        ++degrees[node];
    
    for (auto node : in)
        ++degrees[node];
    
    edgelist = new int*[popsize];
    for (int node = 0; node < popsize; ++node)
        edgelist[node] = new int[degrees[node]];
    
    
    for (int i = 0; i < in.size(); ++i) {
        int node1 = in[i];
        int node2 = out[i];
        edgelist[node1][temp[node1]] = node2;
        edgelist[node2][temp[node2]] = node1;
        ++temp[node1];
        ++temp[node2];
    }
    delete[] temp;
}

// Data structure for a clone in the population. A clone is defined as the set of individual containing the same mutations
class Clone
{
private:
    int clone_id;
    Clone* parent;
    double fitness;
    int count = 0;
    int children = 0;
    int mut_cnt = 0;
    
    vector<int> nodes;
    unordered_map<int, int> index {};
    
public:
    static bool end_of_trial; // Controls cleanup behavior in the destructor

    Clone(int clone_id, double fitness = 1) : clone_id(clone_id), fitness(fitness){ parent = nullptr; }
    
    Clone(int node, int clone_id, double fitness, Clone* parent_ = nullptr) : Clone(clone_id, fitness)
    {
        parent = parent_;
        if(parent != nullptr)
            parent->children++;
        
        nodes.push_back(node);
        mut_cnt = parent->get_mut_cnt() + 1;
        count++;
        index[node] = 0;
    }
    
    ~Clone() 
    {
        if (!end_of_trial && parent != nullptr) {
            parent->children--;
            // if (parent->children == 0) {
            //     delete parent;
            // }
        }
    }
    // return total fitness of all the individuals in the populatoin
    double total_fitness(){return fitness * count;}
    // return fitness of a single individual
    double get_fitness() {return fitness;}
    // return number of mutations in the clone
    double get_mut_cnt() {return mut_cnt;}
    // Get number of individual in this clone
    int get_count() {return count;}
    // return a random node occupied by this clone
    int sample_node(double p) {return nodes[(int)(p * count)]; }
    // return the parent clone.
    Clone* get_parent(){return parent;}
    // get the id of the clone
    int get_id(){return clone_id;}
    // return the number of descendants
    int get_children(){return children;}
    // add a node to this clone
    void add(int);
    // remove a node from tihs clone
    void remove(int);
    // print nodes belonging to this clone
    friend ostream& operator<<(ostream& os, Clone& c)
    {
        for(int node:c.nodes)
            os << node << " ";
        
        os << endl;
        return os;
    }
};

void Clone::add(int node)
{
    if(index.count(node) )
        throw invalid_argument("Node alreadying in the clone!");
    
    nodes.push_back(node);
    index[node] = count;
    count++;
}

void Clone::remove(int node)
{
    if(index.count(node) == 0)
        throw invalid_argument("Node not in the clone!");
    
    count--;
    int idx = index[node];
    
    nodes[idx] = nodes[count];
    index[nodes[count]] = idx;
    
    nodes.pop_back();
    index.erase(node);
}

bool Clone::end_of_trial = false;
// Data structure that keeps track of all the different clones on the graph.
class Population
{
private:
    int N;
    double mu, s;
    double mean_fitness, mean_mut_cnt, var_mut_cnt;
    
    Structure g;
    
    unordered_map<int, Clone*> cid_to_clone;
    unordered_map<int, int> tree;
    int clone_id;
    
    Clone **node_to_clone;
    mt19937 generator;
    uniform_real_distribution<double> rand;
    geometric_distribution<int> rand_geo;
    
    void initialize()
    {
        mean_fitness = 1;
        mean_mut_cnt = 0;
        var_mut_cnt  = 0;
        clone_id = 0;
        
        tree.clear();
        cid_to_clone.clear();
        cid_to_clone[clone_id] = new Clone(clone_id, 1);
        
        node_to_clone = new Clone*[N];
        
        for(int node = 0; node < N; node++)
            cid_to_clone[clone_id]->add(node);
    
        for(int node = 0; node < N; node++)
            node_to_clone[node] = cid_to_clone[clone_id];
    }
    // Follow the ancestry of the input clone. Used for returning the phylogenetic tree of all the clones.
    void tree_trace(Clone* clone, unordered_map<string, int> &edge_map)
    {
        if(clone->get_parent() != nullptr and edge_map.count(to_string(clone->get_id())) == 0)
        {
            Clone* parent = clone->get_parent();
            edge_map[to_string(clone->get_id())] = parent->get_id();
            tree_trace(parent, edge_map);
        }
    }
    
public:
    Population(Structure g, double s = 0, double mu = 0.001): g(g), s(s), mu(mu)
    {
        generator = mt19937((unsigned int)time(NULL));
        rand = uniform_real_distribution<double>(0.0,1.0);
        rand_geo = geometric_distribution<int> (mu);
        
        N = g.popsize;
        initialize();
    }
    
    void clear()
    {
        Clone::end_of_trial = true;
        delete[] node_to_clone;
        for(auto cid : cid_to_clone)
            if (cid.second != nullptr) {
                delete cid.second;  // Only delete if necessary and if not shared
                cid.second = nullptr;
            }
            // delete cid.second;
        Clone::end_of_trial = false;
        initialize();
    }
    // return the birth death node pair
    vector<int> birth_death()
    {
        vector<int> clone_id_vec{};
        vector<double> clone_weight {};
        
        mean_fitness = 0;
        mean_mut_cnt = 0;
        var_mut_cnt = 0;
        for(auto cid : cid_to_clone)
        {
            if (cid.second != nullptr) {
                clone_id_vec.push_back(cid.first);
                clone_weight.push_back(cid.second->total_fitness());
                mean_fitness += cid.second->total_fitness();
                mean_mut_cnt += cid.second->get_mut_cnt() * cid.second->get_count();
                var_mut_cnt += pow(cid.second->get_mut_cnt(), 2) * cid.second->get_count();
            }
            
        }
        
        mean_fitness /= N;
        mean_mut_cnt /= N;
        var_mut_cnt /= N;
        var_mut_cnt -=  pow(mean_mut_cnt, 2);
        
        discrete_distribution<int> discrete(clone_weight.begin(), clone_weight.end());
        Clone* birth_clone = cid_to_clone[clone_id_vec[discrete(generator)]];
        int birth_node = birth_clone->sample_node(rand(generator) );
        
        int death_node = g.edgelist[birth_node][(int)(rand(generator) * g.degrees[birth_node]) ];
        Clone* death_clone = node_to_clone[death_node];
        
        if (death_clone != birth_clone)
        {
            birth_clone->add(death_node);
            death_clone->remove(death_node);

            node_to_clone[death_node] = node_to_clone[birth_node];
            
            if(death_clone->get_count() == 0)
            {
                cid_to_clone.erase(death_clone->get_id());
                if(death_clone->get_children() == 0)
                //    delete death_clone;
                    safeDeleteClone(death_clone->get_id());
            }
        }
        
        return vector<int>{birth_node, death_node};
    }

    void safeDeleteClone(int id) 
    {
        auto it = cid_to_clone.find(id);
        if (it != cid_to_clone.end() && it->second != nullptr) {
            // Check if the clone to be deleted has a parent that needs updating
            Clone* parent = it->second->get_parent();
            delete it->second;  // Delete the clone
            it->second = nullptr;  // Set the pointer in the map to nullptr

            // Now check if the parent needs to be deleted
            if (parent != nullptr && parent->get_children() == 0) {
                // Find the parent in the map and call safeDeleteClone recursively
                for (auto& pair : cid_to_clone) {
                    if (pair.second == parent) {
                        safeDeleteClone(pair.first);
                        break;
                    }
                }
            }
        }
    }

    // mutate selected node
    int mutate(int parent_node)
    {
        Clone* parent_clone = node_to_clone[parent_node];
        clone_id ++;
        
        //cid_to_clone[clone_id] = new Clone(parent_node, clone_id, parent_clone->get_fitness() + s, parent_clone);
        cid_to_clone[clone_id] = new Clone(parent_node, clone_id, parent_clone->get_fitness() * (1 + s), parent_clone);
        tree[clone_id] = parent_clone->get_id();
        
        parent_clone->remove(parent_node);
        node_to_clone[parent_node] = cid_to_clone[clone_id];
        return clone_id;
    }
    // print all clones in the popoulatoin
    friend ostream& operator<<(ostream& os, Population& p)
    {
        for(auto cid : p.cid_to_clone)
            os << cid.first << ":" << cid.second->get_fitness() << endl;
        return os;
    }
    
    // return the edgelist representation of the phylogenetic tree
    unordered_map<string, int> get_tree()
    {
        unordered_map<string, int> edge_map;
        for(auto cid : cid_to_clone)
        {
            edge_map[to_string(cid.second->get_id()) + "\'"] = cid.second->get_id();
            tree_trace(cid.second, edge_map);
        }
        return edge_map;
    }

    vector<int> get_clones()
    {
        vector<int> clone_list(N, 0);
        for(int i = 0; i < N; i++)
            clone_list[i] = node_to_clone[i]->get_id();
        return clone_list;
    }

    vector<int> get_clone_tree()
    {
        vector<int> parentVector(clone_id);
        for(int i = 0; i < clone_id; ++i)
            parentVector[i] = tree[i+1];
        return parentVector;
    }
    
    // return the fitnesses of all the nodes in the populatoin
    vector<double> get_node_fitnesses()
    {
        vector<double> node_to_fit(N, 0);
        for(int i = 0; i < N; i++)
            node_to_fit[i] = node_to_clone[i]->get_fitness();
        return node_to_fit;
    }
    // return time to next mutation event
    int get_next_mutation(){return rand_geo(generator) + 1;} // Gotta add one since c++ geometric distribution starts at 0 instead of 1;
    int number_of_clones(){return static_cast<int>(cid_to_clone.size());}
    int number_of_mutations(){return clone_id;}
    double get_mean_mut_cnt(){return mean_mut_cnt;}
    double get_var_mut_cnt(){return var_mut_cnt;}
};
