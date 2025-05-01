//
//  main.cpp
//  Clonal_interference
//
//  Created by Yang Ping Kuo on 9/9/22.
//

#include <iostream>
#include <iomanip>
#include "clonal.h"

static const int RUNTIME_LIMIT = 24;

int main(int argc, const char * argv[]) {
    Structure g(argv[1]);
    ofstream f_out, f_tree, f_node;
    f_out.open(string(argv[2]) + ".txt");   // txt file containing statistics of simulated mutational trajectories.
    f_tree.open(string(argv[2]) + "_tree.txt"); // txt file containing edgelist representation of the phylogenetic tree.
    f_node.open(string(argv[2]) + "_list.txt"); // final_distribution
    // f_node.open(string(argv[2]) + "_node.txt"); // txt file containing the number of mutation in each node.
    
    double mu = atof(argv[3]);
    int t_e = int(1 / mu);
    double s = atof(argv[4]);
    int trials = atoi(argv[5]);

    int T_max = 1000;
    
    Population p(g, s, mu);
    
    vector<double> mean_traj(T_max + 1, 0);
    vector<double> var_traj(T_max + 1, 0);
    vector<double> trajectory(T_max + 1, 0);
    
    long start = time(NULL), runtime = 0;
    int trial = 0;
    // Loop through all trials of the simulation
    for(trial = 0; trial < trials; trial++)
    {
        int steps = 0;
        p.clear();
        
        int T = 0, t = 0;
        int t_next = p.get_next_mutation();
        // run one trial of the simulation up to time T_max
        while (T <= T_max)
        {
            trajectory[T] = p.get_mean_mut_cnt();
            mean_traj[T] += p.get_mean_mut_cnt();
            var_traj[T] += p.get_var_mut_cnt();
        
            if (T == T_max)
                break;
            while(t < t_e)
            {
                // one Birth-death event
                vector<int> bd_nodes = p.birth_death();
            
                if(t == t_next)
                {
                    p.mutate(bd_nodes[0]);
                    t_next += p.get_next_mutation();
                }
                else if(p.number_of_clones() == 1)
                {
                    t = t_next - 1;
                }
                t++;
                steps++;
            }
            t -= t_e;
            t_next -= t_e;
            T++;
        }

        for(auto n : p.get_clone_tree())
            f_tree << n << "\t";      
        f_tree << endl;  

        for (auto c : p.get_clones()) //number of mutations on each node
            f_node << c << "\t";
        f_node << endl;

        cout << trial << " " << T * t_e + t <<  " "  << steps << " " << T << " " << p.number_of_mutations() << " " << p.get_mean_mut_cnt() << endl;
        cout << endl;
    
        // for (auto c : p.get_node_fitnesses())
        //    f_node << c << "\t";
        // f_node << endl;
        
        // stop simutation if time excesses predefined time
        runtime = (time(NULL) - start);
        if (runtime / 3600 >=  RUNTIME_LIMIT)
        {
            trial++;
            break;
        }
    }
    
    // recording simulation results
    // save single and mean trajectories.
    f_out << "# " << argv[1] << endl;
    f_out << "# mu = " << mu << ", s = " << s <<  ", trials requested = " << trials << ", runtime = " << runtime / 3600 << ":";
    f_out << setfill ('0') << setw (2) << (runtime % 3600) / 60 << ":" << setfill ('0') << setw (2) << (runtime % 60) << endl;
    f_out << "# \"Time\"\t\"Actual trials\"\t\"Mean mutation\"\t\"Average mean mutation\"\t\"Average mutation variance\"" << endl;
    
    for(int T = 0; T <= T_max; T++)
    {
        mean_traj[T] /= trial;
        var_traj[T] /= trial;
        
        f_out << T << "\t";
        f_out << trial << "\t";
        f_out << trajectory[T] << "\t";
        f_out << mean_traj[T] << "\t";
        f_out << var_traj[T] << "\t";
        f_out << endl;
    }
    
    // // save edgelist representation of the phylogenetic tree.
    // for(auto edge : p.get_tree())
    //     f_tree << edge.first << " " << edge.second << endl;
    
    f_out.close();
    f_tree.close();
    f_node.close();
    // f_node.close();
    
    return 0;
}

