/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: Mar 31, 2018 
 *      Author: Ricardo Rios 
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map> 

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

        num_particles = 200;

        Particle part; 
        part.id = 0; 
        part.x = 0;
        part.y = 0; 
        part.theta = 0;
        part.weight = 0;       

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

        int sample_id;
        double sample_x;
        double sample_y; 
        double sample_theta; 

        default_random_engine gen;
      

        for(int i=0; i<num_particles; i++)
        {
           sample_id = i;
           sample_x = dist_x(gen); 
           sample_y = dist_y(gen); 
           sample_theta = dist_theta(gen);
                      

           part.id = sample_id; 
           part.x = sample_x;
           part.y = sample_y; 
           part.theta = sample_theta;
           part.weight = 1.0;       
           particles.push_back(part);   

        }
        
        
        is_initialized = true;         
        
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

        default_random_engine gen;

        double x_new;
        double y_new;
        double theta_new;


        for(int i=0;i<num_particles;i++)
        {

           // if yaw rate is zero, we have to consider other equations.
           // The following web page was very useful: 
           // https://discussions.udacity.com/t/yaw-rate-theta-dot/243585/2

           if (fabs(yaw_rate)>0.0001)
           {
              x_new = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + (yaw_rate*delta_t))-sin(particles[i].theta));
              y_new = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta + (yaw_rate*delta_t)));
              theta_new = particles[i].theta + yaw_rate*delta_t;
           } 

           else 
           {       
              x_new = particles[i].x + velocity*delta_t*cos(particles[i].theta);
              y_new = particles[i].y + velocity*delta_t*sin(particles[i].theta);
              theta_new = particles[i].theta;
           }

          normal_distribution<double> dist_x(x_new,std_pos[0]);
          normal_distribution<double> dist_y(y_new,std_pos[1]);
          normal_distribution<double> dist_theta(theta_new,std_pos[2]);

          particles[i].x = dist_x(gen);
          particles[i].y = dist_y(gen);
          particles[i].theta = dist_theta(gen);

        }

        
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& trans_obser, const Map &map_landmarks){


        int id_min; 
        double min_dist; 
        double dist_temp; 

        for(int i=0; i<trans_obser.size(); i++)
        {
 
           for(int j=0; j<map_landmarks.landmark_list.size(); j++) 
           {
              if ( j == 0)
              {
                 id_min = map_landmarks.landmark_list[j].id_i ;    
                 min_dist = dist(trans_obser[i].x, trans_obser[i].y, 
                                 map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f); 
               }
       
              else
              {

                 dist_temp = dist(trans_obser[i].x, trans_obser[i].y, 
                                  map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f); 

                 if (dist_temp < min_dist) 
                 {
                    id_min = map_landmarks.landmark_list[j].id_i ;    
                    min_dist = dist_temp;                 
                 }
              }          
           }

           trans_obser[i].id = id_min;       

        }

}


double ParticleFilter::compute_probability(const std::vector<LandmarkObs>& trans_obser, const Map &map_landmarks, double std_landmark[]){

        double prob = 1.0; 
        double result = 0.0; 
        int id = -1; 
  
        for(int i=0; i<trans_obser.size(); i++)
        {
           id = trans_obser[i].id; 
           id = id - 1;

          
           result = gaussian_bivariate(trans_obser[i].x, trans_obser[i].y, 
                                       map_landmarks.landmark_list[id].x_f, map_landmarks.landmark_list[id].y_f,
                                       std_landmark[0], std_landmark[1]);

     
  
           prob = prob * result;        

        }

        return prob; 

}



void ParticleFilter::transform_observation(double x_particle, double y_particle, double theta_particle, double x_obs, double y_obs, double &x_map, double &y_map) {
   
        x_map = x_particle + (cos(theta_particle)*x_obs) - (sin(theta_particle)*y_obs); 
        y_map = y_particle + (sin(theta_particle)*x_obs) + (cos(theta_particle)*y_obs); 

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
		   																																								
        std::vector<LandmarkObs> obs_copy = observations;   
        

        for (int i=0; i<num_particles; i++)
        {

           for (int j=0; j<observations.size(); j++)
           {
              transform_observation(particles[i].x, particles[i].y, particles[i].theta, observations[j].x, observations[j].y, 
                                    obs_copy[j].x, obs_copy[j].y);   
                                                    
           }

           dataAssociation(obs_copy, map_landmarks);            
           particles[i].weight = compute_probability(obs_copy, map_landmarks, std_landmark);

        }
        
            
}

void ParticleFilter::resample() {

        
        // Normalizing the weights.   

        double total_weights = 0.0;         

        for (int i=0; i<num_particles; i++)
        {
           total_weights = total_weights + particles[i].weight; 
 
        }        
        


        if (total_weights != 0.0)
        {
           for (int i=0; i<num_particles; i++)
           {
              particles[i].weight = particles[i].weight / total_weights; 
 
           }

        } 


        std::vector<double> w_old;   
        std::vector<Particle> sample_particles; 
    
        for (int i=0; i<num_particles; i++)
        {
           w_old.push_back(particles[i].weight);           
        }
     
        // This web page was very useful: 
        // https://stackoverflow.com/questions/31153610/setting-up-a-discrete-distribution-in-c?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        // Setup the random seed.
        std::random_device rd;
        std::mt19937 gen(rd());  

        std::default_random_engine rand_generator;
        std::discrete_distribution<> d(w_old.begin(), w_old.end());
 

        for (int i=0; i<num_particles; i++)
        {
           sample_particles.push_back(particles[d(gen)]);
        }

        
        for (int i=0; i<num_particles; i++)
        {
           particles[i] = sample_particles[i];
        }




}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
