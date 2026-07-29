//headers and declarations
#include <cmath> // log
//#include <iostream> // cout
#include <gsl/gsl_rng.h> // gsl_rng
#include <gsl/gsl_randist.h> // gsl_ran_gaussian
//#include <limits> //  std::numeric_limits
#include <ctime>  // time(NULL)
#include <string> // string


/* Fixed SEIR model parameters */
// Number of reaction channels:
const int a_no = 10;
// Number of species:
const int s_no = 6;

// Effects Matrix:
const int v[s_no*a_no] = {1,0,0,0,0,0,
						  -1,1,0,0,0,0,
						  0,-1,1,0,0,0,
						  0,0,-1,1,0,1,
						  -1,0,0,0,0,0,
						  0,-1,0,0,0,0,
						  0,0,-1,0,0,0,
						  0,0,0,-1,0,0,
                          0,0,0,0,-1,0,
						  0,0,0,0,1,0};




//struct listing variable model parameters:
struct Parameters{


	// Measles-like defaults:
	double N = 1e6;
	double R0 = 15.;
	double gamm = 1.0/5.0; //recovery rate
	double rho = 1.0/8.0; // exposed to infected rate
	double eta = 1.0/7.; // importation rate
	double mu = 1.0/(75.0*365.0);
	double beta = R0*(gamm + mu)*(rho+mu)/rho; // as mu << rho

	// Seasonality defaults
	double seasonality_amplitude = 0.11; // from Pej's book. Previous runs used 0.03 
	double peak_day = 30;
	double term_time_amplitude = 0.3; //from Pej's book pg 176
	int forcing = 0; // default no forcing

	double Tstart = 50*365;
	double Tdur = 80*365;
	double Tend = Tstart + Tdur;
	double dte=1.0; 	// Output time-step size:


	double initial_uptake = 0.92;
	double final_uptake = 0.70;
	double ramp_start = 40*365;
	double ramp_duration = 0*365;
	double ramp_end = ramp_start + ramp_duration;
	double ramp_slope = (final_uptake - initial_uptake)/ramp_duration;

	double runs = 1; // number of replicates
	double seed = 37427942; // number of replicates
	std::string folder = "./";
	std::string run_name = "./EPI_";

	// Set model parameters using argc and argv inputs
	void set_model_parameters(int c, char **v){
		for(int i =c -1; i > 0; i--){
		    std::string ts = v[i];

		    if(ts.substr(0, 2).compare("N=") == 0){
		        N = std::stof(ts.substr(2));
		    }
		    if(ts.substr(0, 3).compare("R0=") == 0){
		        R0 = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 6).compare("gamma=") == 0){
		        gamm = std::stof(ts.substr(6));
		    }
		    if(ts.substr(0, 4).compare("rho=") == 0){
		        rho = std::stof(ts.substr(4));
		    }
		    if(ts.substr(0, 4).compare("eta=") == 0){
		        eta = std::stof(ts.substr(4));
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
		    if(ts.substr(0, 3).compare("iu=") == 0){
		        initial_uptake = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 3).compare("fu=") == 0){
		        final_uptake = std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 3).compare("rs=") == 0){
		        ramp_start = 365*std::stof(ts.substr(3));
		    }
		    if(ts.substr(0, 3).compare("rd=") == 0){
		        ramp_duration = 365*std::stof(ts.substr(3));
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
		}
	    beta = R0*(gamm + mu)*(rho+mu)/rho; // as mu << rho
	    Tend = Tstart + Tdur;
	    ramp_end = ramp_start + ramp_duration;
		ramp_slope = (final_uptake - initial_uptake)/ramp_duration;
	}


	double seasonal_forcing(double t){
	    return(beta*(1+ seasonality_amplitude*cos(2*M_PI*(t-peak_day)/365.)));
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
	        return(beta*(1.0 + term_time_amplitude*force));
	}
	
	double forcing_function(double t){
		if(forcing == 0) return(beta);
		else if(forcing == 1) return(seasonal_forcing(t));
		else if(forcing == 2) return(term_time_forcing(t));
		else{
			std::cerr << "invalid forcing" << std::endl;
			return(beta);
		}
	}

	//Vaccine uptake function:
	double vaccine_uptake(double t){
	    double vu = initial_uptake;
	    double t_shift = t - Tstart;
	    if( t_shift > ramp_start && t_shift <= ramp_end){
	        vu = initial_uptake + ramp_slope*(t_shift-ramp_start);
	    }
	    else if(t_shift > ramp_end){
	        vu = final_uptake;
	    }
		return(vu);
	}

	//check that beta is correctly reset to initial value...
	void set_initial_conditions(double n[s_no]){
        n[0] = std::min(int(N/R0),int((1.-initial_uptake)*N)) ;
        n[0] = int((1-initial_uptake)*N);
        n[2] = 0;//std::max(int(N*(mu/beta)*((1.-initial_uptake)*R0-1.)), 0);
        n[1] = 0;//int(gamm/rho)*n[2];
        n[4] = int(initial_uptake*N);
        n[3] = N - n[0] - n[1] - n[2] - n[4];
        n[5] = 0;
        //n[s_no] = {S0, E0,I0, N - S0 - E0 -I0- V0, V0,0};

	}

	// Reaction rates: SEIR model with varying vaccine uptake
	void reactions_update(double n[s_no],  double a[a_no], double vaccine_uptake, double beta)
		{
		// Birth of susceptible
		a[0] = (1-vaccine_uptake)*mu*N;
		//Infection of susceptible:
		a[1] = (n[0]/(N-1))*(beta*n[2] + eta) ;
		// Exposed to infectious
		a[2] = rho*n[1];
		//Recovery of infected:
		a[3] = gamm*n[2];
		// Death of susceptible
		a[4] = mu*n[0];
		// Death of exposed
		a[5] = mu*n[1];
		// Death of infectious
		a[6] = mu*n[2];
		// Death of recovered
		a[7] = mu*n[3];
		// Death of vaccinated
		a[8] = mu*n[4];
		// Vaccinated births
		a[9] = vaccine_uptake*mu*N;
		}






    // this code may be used to incorporate future brownian bridge option


	/*double beta;
	double gamm;
	double eta;
	double timmune;
	double N;
	double Tend;
	double runs;
	std::string model;
	double set_beta(double t, double max){return(t/max);}



	//As presently coded the set_beta_* functions actually generate timeseries for R0, not beta
	//hence in get_beta beta_ts is multiplied by gamm.

	std::vector<double> beta_ts;
	double get_beta(double time, double interval){ return beta_ts[int(time/interval)];}

	
	void set_beta_brownian_bridge(double end_time, double interval, gsl_rng * rng, double variance, double curvature){
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
		beta_ts = bb;			
	}
	
	std::vector<double> brownian_bridge(double end_time, double interval, gsl_rng * rng, double variance, double curvature)
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

	void set_beta_restricted_ou(double end_time, double interval, gsl_rng * rng, double drift, double variance, 
		                        double initial_value, double lower_limit, double upper_limit)
	{
		int n = int(end_time/interval);
		double rt_interval = sqrt(interval*variance);
	    std::vector<double> wp(n);
	    int j = 0;   	
	    while(j < n-1){
			j = 0;
			wp[0] = gsl_ran_gaussian(rng, rt_interval/(2*drift)) + initial_value;	
			for(int i = 1; i <n; i++){
				wp[i] = wp[i-1]  +drift*(initial_value-wp[i-1])  + gsl_ran_gaussian(rng, rt_interval);
				if(wp[i] > upper_limit || wp[i] < lower_limit) break;
				j++;
			}
		}
		beta_ts = wp;			
	}



	*/
	
};



