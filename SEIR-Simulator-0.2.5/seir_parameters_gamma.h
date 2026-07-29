//headers and declarations
#include <cmath> // log
//#include <iostream> // cout
#include <gsl/gsl_rng.h> // gsl_rng
#include <gsl/gsl_randist.h> // gsl_ran_gaussian
//#include <limits> //  std::numeric_limits
#include <ctime>  // time(NULL)
#include <string> // string
#include <array>



/* Fixed SEIR model parameters */
// Number of reaction channels:
/*const int a_no = 10;
// Number of species:
const int s_no = 6 +Le + Li;

// Effects Matrix:
const int v[s_no*a_no] = {1,0,0,0,0,0,
						  -1,1,0,0,0,0,

						  0,-1,1,0,0,0,
						  0,0,-1,1,0,1,

						  -1,0,0,0,0,0,
						  0,0,0,-1,0,0,
                          0,0,0,0,-1,0,
						  0,0,0,0,1,0,						  
						  0,-1,0,0,0,0,
						  0,0,-1,0,0,0,};

*/

// Number of exposed and infectious classes
const int Le = 1;
const int Li = 1;
// Number of reaction channels:
const int a_no = 6 + 2*Le + 2*Li;
// Number of species:
const int s_no = 6 +Le + Li;


//struct listing variable model parameters:
struct Parameters{


	// Measles-like defaults:
	double gamm = 1.0/5.0; //recovery rate
	double rho = 1.0/8.0; // exposed to infected rate
	double mu = 1.0/(75.0*365.0);

	// Seasonality defaults
	double seasonality_amplitude = 0.11; // from Pej's book. Previous runs used 0.03 
	double peak_day = 30;
	double term_time_amplitude = 0.3; //from Pej's book pg 176
	int forcing = 0; // default no forcing

	double Tstart = 50*365;
	double Tdur = 80*365;
	double Tend = Tstart + Tdur;
	double dte=1.0; 	// Output time-step size:

    // Reporting process
	double rep_dispersion = 1;


	double v_i = 0.70;
	double v_f = 0.70;
	double v_rs = 40*365;
	double v_rd = 0*365;


	double eta_i = 1.0/7.;
	double eta_f = 1.0/7.;
	double eta_rs = 40*365;
	double eta_rd = 0*365;

	double rp_i = 0.1;
	double rp_f = 0.1;
	double rp_rs = 40*365;
	double rp_rd = 0*365;

	double N_i = 1e6;
	double N_f = 1e6;
	double N_rs = 40*365;
	double N_rd = 0*365;

	double R0_i = 15;
	double R0_f = 15;
	double R0_rs = 40*365;
	double R0_rd = 0*365;

	double runs = 1; // number of replicates
	double seed = 37427942; // rng seed
	std::string folder = "./";
	std::string run_name = "./epi_data";

	int R0_ramp = 1; // options: linear, bb, ou, constant
	double bb_var =0.00001;
	double bb_a =1;
	double R0_ou_drift =0.1;
    double R0_ou_var=0.00001;
    double R0_lower_limit=0;
    double R0_upper_limit=1;






	// Set model parameters using argc and argv inputs
	void set_model_parameters(int c, char **v){
		for(int i =c -1; i > 0; i--){
		    std::string ts = v[i];

		    if(ts.substr(0, 6).compare("gamma=") == 0){
		        gamm = std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 4).compare("rho=") == 0){
		        rho = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 3).compare("mu=") == 0){
		        mu = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 3).compare("sa=") == 0){
		        seasonality_amplitude = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 4).compare("tta=") == 0){
		        term_time_amplitude = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 3).compare("pd=") == 0){
		        peak_day = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 6).compare("force=") == 0){
    			if(ts.substr(6,10).compare("none") == 0){ forcing = 0;}
		    	else if(ts.substr(6,14).compare("seasonal") == 0){ forcing = 1;}
		    	else if(ts.substr(6,10).compare("term") == 0){forcing = 2;}
		    	else{std::cerr << "Invalid argument to force. Default used: none" << std::endl;}
		    }
		    if(ts.substr(0, 6).compare("relax=") == 0){
		        Tstart = 365*std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 2).compare("T=") == 0){
		        Tdur= 365*std::stof(ts.substr(2));
		    }
		    if(ts.substr(0, 3).compare("ts=") == 0){
		        dte= std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 5).compare("runs=") == 0){
		        runs = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("seed=") == 0){
		        seed = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 7).compare("folder=") == 0){
		        folder = ts.substr(7);
		    }
            if(ts.substr(0, 3).compare("rn=") == 0){
		        run_name = ts.substr(3);
		    }
		    if(ts.substr(0, 5).compare("rdis=") == 0){
		        rep_dispersion = std::stof(ts.substr(5));
		    }

		    if(ts.substr(0, 4).compare("v_i=") == 0){
		        v_i = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 4).compare("v_f=") == 0){
		        v_f = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 5).compare("v_rs=") == 0){
		        v_rs = 365*std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("v_rd=") == 0){
		        v_rd = 365*std::stof(ts.substr(5));
		    }


		    if(ts.substr(0, 6).compare("eta_i=") == 0){
		        eta_i = std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 6).compare("eta_f=") == 0){
		        eta_f = std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 7).compare("eta_rs=") == 0){
		        eta_rs = 365*std::stof(ts.substr(7));
		    }
		    if(ts.substr(0, 7).compare("eta_rd=") == 0){
		        eta_rd = 365*std::stof(ts.substr(7));
		    }

		    if(ts.substr(0, 5).compare("rp_i=") == 0){
		        rp_i = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("rp_f=") == 0){
		        rp_f = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 6).compare("rp_rs=") == 0){
		        rp_rs = 365*std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 6).compare("rp_rd=") == 0){
		        rp_rd = 365*std::stof(ts.substr(6));
		    }

		    if(ts.substr(0, 4).compare("N_i=") == 0){
		        N_i = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 4).compare("N_f=") == 0){
		        N_f = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 5).compare("N_rs=") == 0){
		        N_rs = 365*std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("N_rd=") == 0){
		        N_rd = 365*std::stof(ts.substr(5));
		    }

		    if(ts.substr(0, 5).compare("R0_i=") == 0){
		        R0_i = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("R0_f=") == 0){
		        R0_f = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 6).compare("R0_rs=") == 0){
		        R0_rs = 365*std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 6).compare("R0_rd=") == 0){
		        R0_rd = 365*std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 8).compare("R0_ramp=") == 0){
		        if(ts.substr(8,14).compare("linear") == 0){ R0_ramp = 0;}
		    	else if(ts.substr(8,13).compare("fixed") == 0){ R0_ramp = 1;}
		    	else if(ts.substr(8,10).compare("bb") == 0){R0_ramp = 2;}
		        else if(ts.substr(8,10).compare("ou") == 0){R0_ramp = 3;}
		    	else{std::cerr << "Invalid argument to force. Default used: linear" << std::endl;}
		    }
            if(ts.substr(0, 5).compare("bb_a=") == 0){
		        bb_a = std::stof(ts.substr(5));
		    }
		    if(ts.substr(0, 5).compare("bb_v=") == 0){
		        bb_var = std::stof(ts.substr(5));
		    }
            if(ts.substr(0, 5).compare("ou_d=") == 0){
		       R0_ou_drift = std::stof(ts.substr(5));
		    }
            if(ts.substr(0, 5).compare("ou_v=") == 0){
		       R0_ou_var = std::stof(ts.substr(5));
		    }
            if(ts.substr(0, 5).compare("R0_l=") == 0){
		       R0_lower_limit = std::stof(ts.substr(5));
		    }
            if(ts.substr(0, 5).compare("R0_u=") == 0){
		       R0_upper_limit = std::stof(ts.substr(5));
		    }



		}

	    Tend = Tstart + Tdur;
	}
	std::vector<double> beta_ts;


	double seasonal_forcing(double t){
	    return(1+ seasonality_amplitude*cos(2*M_PI*(t-peak_day)/365.));
	}

	// Term time forcing using uk terms
	double term_time_forcing(double t){

	        int day = int(t) % 365;
	        double force;
	        double coef = (11.+16.+52.+7.+10.)/365.;
	        if (day <= 10){
	            force = -1.0;  // new year break January 1 - January 10
	        }
	        else if ((day >= 100) && (day <= 115)){
	             force = -1.0; //spring break April 10 - 25
	        }
	        else if((day >= 200) && (day <= 251)){
	             force = -1.0;  // summer June 19 - September 8
	        }
	        else if((day >= 301) && (day <= 307)){
	            force = -1.0;  // autumn break October 28 - November 3
	        }
	        else if((day >= 356) and (day <= 365)){
	            force = -1.0;  //winter break December 21 - December 31
	        }
	        else{
	            force = coef/(1.0 - coef);
	        }
	        return(1.0 + term_time_amplitude*force);
	}
	
	double forcing_function(double t){
		if(forcing == 0) return(1);
		else if(forcing == 1) return(seasonal_forcing(t));
		else if(forcing == 2) return(term_time_forcing(t));
		else{
			std::cerr << "invalid forcing" << std::endl;
			return(1);
		}
	}

//
    double ramp_function(double t, double duration, double initial_time,
                         double initial_value, double final_value){

 	    double ramp_slope = (final_value - initial_value)/duration;
 	    double final_time = initial_time + duration;

	    double vu = initial_value;
	    double t_shift = t - Tstart;
	    if( t_shift > initial_time && t_shift <= final_time){
	        vu = initial_value + ramp_slope*(t_shift-initial_time);
	    }
	    else if(t_shift > final_time){
	        vu = final_value;
	    }
		return(vu);
    }

	//Vaccine uptake function:
	double vaccine_uptake(double t){
	    return(ramp_function(t, v_rd,v_rs,v_i, v_f));
	   }


	double eta_function(double t){
	    return(ramp_function(t,eta_rd,eta_rs,eta_i,eta_f));
	}

	double rep_prob_function(double t){
	    return(ramp_function(t,rp_rd,rp_rs,rp_i,rp_f));
	}


	double pop_function(double t){
	    return(ramp_function(t,N_rd,N_rs,N_i,N_f));
	}


	double R0_function(double t){
	    double R0;
        if(R0_ramp == 0){
            R0 = ramp_function(t,R0_rd,R0_rs,R0_i,R0_f);
        }
        if(R0_ramp == 1){
            R0 = R0_i;
        }
        if(R0_ramp == 2){
            if(t < Tstart + R0_rs){
                R0 = R0_i;
            }
            else if(t > Tstart +R0_rs + R0_rd){
                R0 = R0_f;
            }
            else{
                R0 = beta_ts[int((t-R0_rs-Tstart)/dte)];
            }
        }
        if(R0_ramp == 3){
            if(t < Tstart + R0_rs){
                R0 = R0_i;
            }
            else if(t > Tstart +R0_rs + R0_rd){
                R0 = R0_i;
            }
            else{
                R0 = beta_ts[int((t-R0_rs-Tstart)/dte)];
            }
        }

	    return(R0*forcing_function(t));
	}

	double beta_function(double t){
	    return(((gamm + mu)*(rho+mu)/rho)*R0_function(t));
	}




    int reported_cases(const gsl_rng * r,double cases, double t){
        //Note: parameterisation of gsl negative binomial is that it returns
        //number of failures before n successes, not vice versa
        double nb_p = rep_dispersion/(rep_prob_function(t)*cases+rep_dispersion);
        return(gsl_ran_negative_binomial(r, nb_p,
                                  rep_dispersion));
    }






	//check that beta is correctly reset to initial value...
	void set_initial_conditions(double n[s_no]){
        //n[0] = std::min(int(N/R0),int((1.-v_i)*N)) ;
        n[0] = int((1-v_i)*N_i);
        double I_init = 0;//std::max(int(N*(mu/beta)*((1.-v_i)*R0-1.)), 0);
        double E_init = 0;//int(gamm/rho)*I_init;
        n[1] = 0;
        n[2] = 0;

        // Initial exposed
        for(int i = 6; i <= 5+Le; i++){
        	n[i] = int(E_init/Le);
        	n[1] += n[i];
        }
        // Initial infected
        for(int i = 6+Le; i <=5+Le+Li; i++){
        	n[i] = int(I_init/Li);
        	n[2] += n[i];
        }
        n[4] = int(v_i*N_i);
        n[3] = N_i - n[0] - n[1] - n[2] - n[4];
        n[5] = 0;

        //n[s_no] = {S0, E0,I0, N - S0 - E0 -I0- V0, V0,0};

	}

	// Reaction rates: SEIR model with varying vaccine uptake
	void reactions_update(double n[s_no],  double a[a_no], double t){ // double vaccine_uptake, double beta){
		// Birth of susceptible
		a[0] = (1-vaccine_uptake(t))*mu*pop_function(t);
		//Infection of susceptible:
		a[1] = (n[0]/(n[0]+n[1]+n[2]+n[3]+n[4]))*(beta_function(t)*n[2]
		       + eta_function(t));
		// Death of susceptible
		a[2] = mu*n[0];
		// Death of recovered
		a[3] = mu*n[3];
		// Death of vaccinated
		a[4] = mu*n[4];
		// Vaccinated births
		a[5] = vaccine_uptake(t)*mu*pop_function(t);


		// Exposed to infectious
        for(int i = 1; i <=Le; i++){
			a[5+i] = rho*n[5+i]*Le;
			a[5+i + Le] = mu*n[5+i];
		}

		// Infectious to recovered
		for(int i = 1; i <=Li; i++){
			a[5+i+2*Le] = gamm*n[5+Le+i]*Li;
			a[5+i + 2*Le + Li] = mu*n[5+Le+i];
		}
	}

	std::array<std::array<int, s_no>, a_no>  set_v(){
		std::array<std::array<int, s_no>, a_no> v;
			v[0] = {1,0,0,0,0,0};
			v[1] = {-1,1,0,0,0,0};
			v[1][6] = 1;
			v[2] = {-1,0,0,0,0,0};
			v[3]= {0,0,0,-1,0,0};
			v[4]= {0,0,0,0,-1,0};
			v[5]= {0,0,0,0,1,0};

			// Exposed stages
		    for(int i = 1; i <Le; i++){
				//v[5+i] = {0,0,0,0,0,0};
				v[5+i][5+i] = -1;
				v[5+i][5+i+1] = 1;
			}

			// Exposed to infectious
			v[5+Le] = {0,-1,1,0,0,0};
			v[5+Le][5+Le] = -1;
			v[5+Le][6+Le] = 1;

			// Exposed death
			for(int i = 1; i <=Le; i++){
				v[5+i + Le] = {0,-1,0,0,0,0};
				v[5+i + Le][5+i] = -1;
			}
			// Infectious stages
		    for(int i = 1; i <Li; i++){
				//v[5+2*Le+i] = {0,0,0,0,0,0};
				v[5+2*Le+i][5+Le+i] = -1;
				v[5+2*Le+i][5+Le+i+1] = 1;
			}
			// Recovery
			v[5+2*Le+Li] = {0,0,-1,1,0,1};
			v[5+2*Le+Li][5+Le+Li] = -1;

			// Death of infectious
		    for(int i = 1; i <=Li; i++){
				v[5 +2*Le+i + Li] = {0,0,-1,0,0,0};
				v[5+2*Le +i + Li][5+Le+i] = -1;
			}

			return  v;
	

    }



	//As presently coded the set_beta_* functions actually generate timeseries for R0, not beta
	//hence in get_beta beta_ts is multiplied by gamm.
    void set_beta(gsl_rng * rng){
        if(R0_ramp == 2){
            set_beta_brownian_bridge(rng);
        }
        else if(R0_ramp == 3){
            set_beta_restricted_ou(rng);
        }
    }


	
	void set_beta_brownian_bridge(gsl_rng * rng){
		int n = int(R0_rd/dte);
		double rt_interval = sqrt(dte*bb_var);
	    	std::vector<double> wp(n);
	    	std::vector<double> bb(n);
	    	int j = 0;
	    	
	    	while(j < n-1){
	    		j = 0;
			    wp[0] = 0;
	
                for(int i = 1; i <n; i++){
                    wp[i] = wp[i-1] + gsl_ran_gaussian(rng, rt_interval);
                }

                for(int i = 0; i <n; i++){
                    bb[i] =  R0_i + (R0_f-R0_i)*(wp[i] +pow(dte*(i/R0_rd),bb_a) - (dte*i/R0_rd)*wp[n-1]);
                    if(bb[i] > R0_upper_limit || bb[i] < R0_lower_limit) break;
                    j++;
                }
		}
		beta_ts = bb;			
	}

	void set_beta_restricted_ou(gsl_rng * rng)
	{
		int n = int(R0_rd/dte);
		double rt_interval = sqrt(dte*R0_ou_var);
	    std::vector<double> wp(n);
	    int j = 0;   	
	    while(j < n-1){
			j = 0;
			// initialise OU process at the initial value
			wp[0] = R0_i; //  gsl_ran_gaussian(rng, rt_interval/(2*R0_ou_drift)) +
			for(int i = 1; i <n; i++){
				wp[i] = wp[i-1]  +R0_ou_drift*(R0_i-wp[i-1])  + gsl_ran_gaussian(rng, rt_interval);
				if(wp[i] > R0_upper_limit || wp[i] < R0_lower_limit) break;
				j++;
			}
		}
		beta_ts = wp;			
	}




	/*

	double vaccine_uptake(double t){
	    double vu = v_i;
	    double t_shift = t - Tstart;
	    if( t_shift > ramp_start && t_shift <= ramp_end){
	        vu = v_i + ramp_slope*(t_shift-ramp_start);
	    }
	    else if(t_shift > ramp_end){
	        vu = v_i;
	    }
		return(vu);
	}
*/
	//


};




/**



	std::vector<double> brownian_bridge(double dtetime, double interval, gsl_rng * rng, double variance, double curvature)
	{
		int n = int(end_time/interval);
		double rt_interval = sqrt(interval*variance);
	    	std::vector<double> wp(n);
	    	std::vector<double> bb(n);
	    	int j = 0;

	    	while(j < n-1){
	    		j = 0;
				wp[0] = 0;

				for(int i = 1; i <n; i++){
					wp[i] = wp[i-1] + gsl_ran_gaussian(rng, rt_interval);
				}

				for(int i = 0; i <n; i++){
					bb[i] = wp[i] +pow(interval*i/end_time,curvature) - (interval*i/end_time)*wp[n-1];
					if(bb[i] < 0 || bb[i] > 1) break;
					j++;
				}
			}
		return bb;
	}

	void set_beta_restricted_bm(double end_time, double interval, gsl_rng * rng, double variance,
					  double initial_value, double lower_limit, double upper_limit)
	{
		int n = int(end_time/interval);
		double rt_interval = sqrt(interval*variance);
	    std::vector<double> wp(n);
	    int j = 0;
		while(j < n-1){
			j = 0;
			wp[0] = initial_value;
			for(int i = 1; i <n; i++){
				wp[i] = wp[i-1] + gsl_ran_gaussian(rng, rt_interval);
				if(wp[i] > upper_limit || wp[i] < lower_limit) break;
				j++;
			}
		}
		beta_ts = wp;
	}

	*/
