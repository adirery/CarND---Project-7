/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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

#include "particle_filter.h"

using namespace std;

// creating random engine to be used across multiple methods
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Initialize number of particles 
	num_particles = 100;

	// set length for particles & weights vector
	particles.resize(num_particles);
	weights.resize(num_particles);

	// Initialize gaussian noise for x, y, theta
	normal_distribution<double> x_noise(0.0, std[0]);
	normal_distribution<double> y_noise(0.0, std[1]);
	normal_distribution<double> theta_noise(0.0, std[2]);

	// Initialize particles (values from GPS & weight = 1)
	for (unsigned int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].x = x;
		particles[i].y = y;
		particles[i].theta = theta;
		particles[i].weight = 1.0 / num_particles;

		// add noise for x, y, theta
		particles[i].x += x_noise(gen);
		particles[i].y += y_noise(gen);
		particles[i].theta += theta_noise(gen);

		// initialize weights vector
		weights[i] = particles[i].weight;
	}
	is_initialized = true;
	
	//TODO: Remove - for debugging only
	cout << "Particle initialized " << particles[0].id << " " << particles[0].x << " " << particles[0].y << " " << particles[0].theta << " " << particles[0].weight << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Initialize gaussian PROCESS noise for x, y, theta
	normal_distribution<double> x_process_noise(0.0, std_pos[0]);
	normal_distribution<double> y_process_noise(0.0, std_pos[1]);
	normal_distribution<double> theta_process_noise(0.0, std_pos[2]);

	// Predict each particle's next state
	for (unsigned int i = 0; i < num_particles; i++) {

		// if no yaw rate use "simple formula"
		if (fabs(yaw_rate) < 0.001) {
			particles[i].x += cos(particles[i].theta) * velocity * delta_t;
			particles[i].y += sin(particles[i].theta) * velocity * delta_t;
		}
		// if yaw rate use more complex formula
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add process noise
		particles[i].x += x_process_noise(gen);
		particles[i].y += y_process_noise(gen);
		particles[i].theta += theta_process_noise(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	vector<LandmarkObs> result;

	// Go through each observation to find related landmark ID
	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];

		// initialize min distance to max possible distance & map index
		double min_dist = INFINITY;
		int map_index = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs prediction = predicted[j];

			// get distance between observation & prediction
			double dist_obs_pred = (observation.x - prediction.x) * (observation.x - prediction.x) + (observation.y - prediction.y) * (observation.y - prediction.y);

			// find the nearest landmark
			if (dist_obs_pred < min_dist) {
				min_dist = dist_obs_pred;
				map_index = j; 
			}
		}
		result.push_back(LandmarkObs{map_index, observation.x, observation.y});
	}
	observations = result;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
	// 1. Create predictions vector based on map_landmark list
	// 2. Create observations vector in "particle space"
	// 3. Associate predictions with transformed observation
	// 4. Calculate particle weight based on pred vs obs "quality"

	for (unsigned int i = 0; i < num_particles; i++) {
		auto p = particles[i];

		// create landmarks vector "in range" based on map_landmark list
		vector<LandmarkObs> map_landmarks_in_range;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			// filter out landmarks "outside" of sensor range
			if (((landmark_x - p.x) * (landmark_x - p.x) + (landmark_y - p.y) * (landmark_y - p.y)) <= sensor_range * sensor_range) {
				map_landmarks_in_range.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		// create transformed observation vector
		vector<LandmarkObs> observations_in_map_coordinates;
		for (unsigned int j = 0; j < observations.size(); j++) {
			float transformed_x = p.x + observations[j].x * cos(p.theta) - observations[j].y * sin(p.theta);
			float transformed_y = p.y + observations[j].x * sin(p.theta) + observations[j].y * cos(p.theta);
			int transformed_id = observations[j].id;

			observations_in_map_coordinates.push_back(LandmarkObs{ transformed_id, transformed_x, transformed_y });
		}
		
		// associate observations with predictions
		dataAssociation(map_landmarks_in_range, observations_in_map_coordinates);

		// reset weights
		double new_weight = 1.0f;

		// calculate new particle weight based on transformed observation vs prediction quality
		// calculate weight using multivariant gaussian
		double std_x = std_landmark[0];
		double cov_x = std_x * std_x;

		double std_y = std_landmark[1];
		double cov_y = std_y * std_y;

		double normalizer = 2.0 * M_PI * std_x * std_y;

		for (unsigned int j = 0; j < observations_in_map_coordinates.size(); j++) {
			// calculate weight using multivariant gaussian
			LandmarkObs obs = observations_in_map_coordinates[j];
			LandmarkObs pred = map_landmarks_in_range[obs.id];

			double dx = obs.x - pred.x;
			double dy = obs.y - pred.y;

			new_weight *= exp(-0.5 * (dx * dx / (cov_x) + dy * dy / (cov_y))) / normalizer;
		}

		particles[i].weight = new_weight;
		weights[i] = new_weight;
	}
	/*
	double sum = 0.0;
	for (unsigned int i = 0; i < weights.size(); i++) {
		sum += weights[i];
	}
	//std::cout << "weights sum before normalizing " << sum << std::endl;
	for (unsigned int i = 0; i < particles.size(); i++) {
		const double weight = particles[i].weight / sum;
		particles[i].weight = weight;
		weights[i] = weight;
	}

	sum = 0.0;
	for (unsigned int i = 0; i < particles.size(); i++) {
		sum += particles[i].weight;
	}
	//std::cout << "particles weights sum after normalization " << sum << std::endl;
	*/
}

void ParticleFilter::resample() {
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> new_particles;
	vector<double> new_weights;

	for (unsigned i = 0; i < num_particles; i++)
	{
		new_particles.push_back(particles[distribution(gen)]);
		new_weights.push_back(1.0);
	}
	particles = new_particles;
	weights = new_weights;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
