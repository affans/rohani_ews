/*ABOUT: This is an implementation of the MNRM as found in Anderson (2008).
*/

//headers and declarations
#include <cmath> // log
#include <fstream> // ofstream
#include <iostream> // cout
#include <gsl/gsl_rng.h> // gsl_rng
#include <limits> //  std::numeric_limits
#include <ctime>  // time(NULL)
#include <vector> // std::vector
#include "./seir_parameters_gamma.h"
#include <string>
#include <array>

// Need to fix this so model doesn't need recompiling for each parameter combination
const std::string folder = "./";
std::ofstream out;

int main(int argc, char **argv)
{
    //Initialise model parameters
    Parameters par;
    par.set_model_parameters(argc, argv);

    // v is the stoichiometry matrix v[k][i] = change in species i when reaction k fires
    // the set_v() function builds this programmatically
    std::array<std::array<int, s_no>, a_no> v = par.set_v();
	
    // random number generation
    srand(par.seed);
	gsl_rng * rng = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set (rng, rand());

    std::string filename = par.folder + "/"+ par.run_name + ".csv";
    out.open(filename.c_str());

    // Output csv header
    out << "time" << "," << "S" << "," << "E" << ","  << "I" << ","
        << "R" << ","  << "V" << ","  << "cases" << ","
        << "reported_cases" << ","
        << "reporting_frac" << ","
        << "uptake" << "," << "R0" << ","<< "eta" << "," << "run" << std::endl;

    for(int run = 1; run <= par.runs; run++){
        /**** algorithm variables: ****/

        // Per-run initialization
        double t = 0; // current simulated (real) time
        double a[a_no]; // propensity vector, filled each step by reactions_update()
        double n[s_no]; // species/state vector so e.g, n[2] is the number of infected individuals
        int nu; // 	index of the reaction chosen to fire next
        double dt; // 	waiting time until the next reaction
        double te = 0;

        par.set_beta(rng); // linear, BB, or OU based on the R0_ramp parameter
        par.set_initial_conditions(n); // set initial conditions for the state vector, SEIRV

        // Internal Poisson processes, internal clocks, next-firing times:
        // The three per-reaction arrays: Pk: next Poisson threshold,  Tk: internal clock,  Dk: candidate wait time.
        // P[a_no]: next unrealized arrival ("internal time") of channel k's unit-rate Poisson process
        // T[a_no]: channel k's internal clock, i.e. integrated propensity so far
        // D[a_no]: candidate waiting time for channel k
        double P[a_no], T[a_no], D[a_no];
        
        //Next-reaction method algorithm: implementation of the MNRM as found in Anderson (2008)

        //Initialisation of internal Poisson processes and internal clocks:
        for(int k = 0; k < a_no; k++){
            // Initialize each reaction's Poisson process with its first arrival time
            P[k] = -log((double)rand()/RAND_MAX); // this is P[k] ~ Exp(1)
            T[k] = 0;
        }

        while(t < par.Tend){ // Each iteration of while(t < par.Tend) is one reaction event — one jump of the CTMC.
            
            // Updating the reaction rates, fills a[k] from the current state
            par.reactions_update(n, a, t); 

            // Calculating which reaction fires next:
            dt = par.Tend;                         //  initialize Δ to a huge value
            for(int k = 0; k < a_no; k++){
                D[k] = (P[k] - T[k])/a[k];         // Δ_k = (P_k − T_k) / a_k
                if(D[k] <= dt && a[k] > 10e-20){   // keep the minimum, skip zero-rate reactions
                    dt = D[k]; nu = k;             // Δ = Δ_μ, μ = argmin
                }
            }

            // Output (while loop incase time-to-next reaction (dt) is larger than one timestep)
            // Note: present implementation of time varying parameters assumes rate of change << dte
            while(t+dt > te){
                if(te >= par.Tstart){
                     out << te-par.Tstart<< ",";
                     for(int i=0; i < s_no - Li - Le; i++){
                        out << n[i] << ",";
                     }
                     out << par.reported_cases(rng, n[s_no - Li - Le-1], t) << ","
                         << par.rep_prob_function(t) << ","
                         << par.vaccine_uptake(t) << ","
                         << par.R0_function(t) << ","
                         << par.eta_function(t) << ","
                         << run << std::endl;
                }
                n[5] = 0;
                te += par.dte;
            }

            // Fire the reaction: this advances the time, Replenish the fired reaction's Poisson process, Apply the stoichiometry
            // Updating the internal Possion process and system state according to the reaction with fired:
            t = t+ dt;   // t ← t + Δ            
            P[nu] -= log(gsl_rng_uniform_pos (rng) );  // P_μ ← P_μ + Exp(1)
            for(int i = 0; i < s_no; i++) n[i] += v[nu][i];  // n ← n + ν_μ


        // Internal clocks are updated:
            for(int k = 0; k < a_no; k++){
                T[k] += dt*a[k];
            }
        }
        std::cerr << "Run " << run << std::endl;
    }

out.close();
return(0);
}


////////

