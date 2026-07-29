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
    std::array<std::array<int, s_no>, a_no> v = par.set_v();


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
        double t = 0;
        double a[a_no];
        double n[s_no];
        int nu;
        double dt;
        double te = 0;

        par.set_beta(rng);
        par.set_initial_conditions(n);


        // Internal Poisson processes, internal clocks, next-firing times:
        double P[a_no], T[a_no], D[a_no];

  
        //Next-reaction method algorithm

        //Initialisation of internal Poisson processes and internal clocks:
        for(int k = 0; k < a_no; k++){
            P[k] = -log((double)rand()/RAND_MAX);
            T[k] = 0;
        }

        while(t < par.Tend){
            // Updating the reaction rates:
            par.reactions_update(n,  a, t);
            // Calculating which reaction fires next:
            dt = par.Tend;
            for(int k = 0; k < a_no; k++){
                D[k] = (P[k] - T[k])/a[k];
                if(D[k] <= dt && a[k] > 10e-20){
                    dt = D[k]; nu = k;
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

            // Updating the internal Possion process and system state according to the reaction with fired:
            t = t+ dt;
            P[nu] -= log(gsl_rng_uniform_pos (rng) );
            for(int i = 0; i < s_no; i++) n[i] += v[nu][i];


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

