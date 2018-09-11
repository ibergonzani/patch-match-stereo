#include <pm.h>
#include <random>

#define WINDOW_SIZE 35
#define MAX_DISPARITY 60
#define PLANE_PENALTY 120


template <typename T>
Matrix2D<T>::Matrix2D() {}

template <typename T>
Matrix2D<T>::Matrix2D(unsigned int rows, unsigned int cols) 
				: rows(rows), cols(cols), data(rows, std::vector<T>(cols)) { }

template <typename T>
Matrix2D<T>::Matrix2D(unsigned int rows, unsigned int cols, const T &def)
				: rows(rows), cols(cols), data(rows, std::vector<T>(cols, def)) { }

template <typename T>
inline T& Matrix2D<T>::operator()(unsigned int row, unsigned int col)
{
	return data[row][col];
}

template <typename T>
inline const T& Matrix2D<T>::operator()(unsigned int row, unsigned int col) const
{
	return data[row][col];
}



Plane::Plane() {}


Plane::Plane(cv::Vec3f point, cv::Vec3f normal) : point(point), normal(normal)
{
	float a = -normal[0] / normal[2];
	float b = -normal[1] / normal[2];
	float c = cv::sum(normal.mul(point))[0] / normal[2];
	coeff = cv::Vec3f(a, b, c);
}

inline float Plane::operator[](int idx) const { return coeff[idx]; }

inline cv::Vec3f Plane::operator()() { return coeff; }

inline cv::Vec3f Plane::getPoint() { return point; }

inline cv::Vec3f Plane::getNormal() { return normal; }

inline cv::Vec3f Plane::getCoeff() { return coeff; }


Plane Plane::viewTransform(int x, int y, int sign, int& qx, int &qy)
{
	float z = coeff[0] * x + coeff[1] * y + coeff[2];
	qx = x + sign * z;
	qy = y;
	
	cv::Vec3f p(qx, qy, z);
	return Plane(p, this->normal);
}


namespace pm
{
	
	void compute_greyscale_gradient(const::cv::Mat3b &frame, cv::Mat2f &grad)
	{
		int scale = 1, delta = 0;
		cv::Mat gray, x_grad, y_grad;

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::Sobel(gray, x_grad, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
		cv::Sobel(gray, y_grad, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
		x_grad = x_grad / 8.f;
		y_grad = y_grad / 8.f;
		
		//#pragma omp parallel for
		for (int y = 0; y < frame.rows; ++y) {
			for (int x = 0; x < frame.cols; ++x) {
				grad(y, x)[0] = x_grad.at<float>(y, x);
				grad(y, x)[1] = y_grad.at<float>(y, x);
			}
		}
	}

	
	
	PatchMatch::PatchMatch(float alpha, float gamma, float tau_c, float tau_g)
	: alpha(alpha), gamma(gamma), tau_c(tau_c), tau_g(tau_g) { }
	
	

	float PatchMatch::dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq, const cv::Vec2f &pg, const cv::Vec2f &qg)
	{
		float cost_c = cv::norm(pp - qq, cv::NORM_L1);
		float cost_g = cv::norm(pg - qg, cv::NORM_L1);
		cost_c = std::min(cost_c, this->tau_c);
		cost_g = std::min(cost_g, this->tau_g);
		return (1 - this->alpha) * cost_c + this->alpha * cost_g;
	}

	
	
	// aggregated matchig cost of a plane for a pixel
	float PatchMatch::plane_match_cost(const Plane &p, int cx, int cy, int ws, int cpv)
	{
		int sign  = -1 + 2 * cpv;
		
		float cost = 0;
		int half = ws / 2;
		
		const cv::Mat3b& f1 = views[cpv];
		const cv::Mat3b& f2 = views[1-cpv];
		const cv::Mat2f& g1 = grads[cpv];
		const cv::Mat2f& g2 = grads[1-cpv];
		const cv::Mat& w1   = weigs[cpv];
		
		for(int x = cx - half; x <= cx + half; ++x)
		{
			for(int y = cy - half; y <= cy + half; ++y)
			{
				if(!inside(x, y, 0, 0, f1.cols, f1.rows))
					continue;
				
				//computing disparity
				float dsp = disparity(x, y, p);
				
				if(dsp < 0 || dsp > MAX_DISPARITY)
				{
					cost += PLANE_PENALTY;
				}
				else
				{
					// find matching point in other view
					float match = x + sign * dsp;
					int x_match = (int)match;
					
					float wm = 1 - (match - x_match);
					
					if(x_match > f1.cols - 2)
						x_match = f1.cols - 2;
					if(x_match < 0)
						x_match = 0;
					
					// and evaluating its color and gradinet (averaged)
					cv::Vec3b mcolo = vecAverage(f2(y, x_match), f2(y, x_match+1), wm);
					cv::Vec2b mgrad = vecAverage(g2(y, x_match), g2(y, x_match+1), wm);
					
					float w = w1.at<float>(cv::Vec<int,4>{cy, cx, y -cy +half, x -cx +half});
					cost += w * dissimilarity(f1(y, x), mcolo, g1(y, x), mgrad);
				}
			}
		}
		
		return cost;
	}
	
	
	
	void PatchMatch::precompute_pixels_weights(const cv::Mat3b &frame, cv::Mat &weights, int ws)
	{
		int half = ws / 2;
		
		#pragma omp parallel for
		for(int cx=0; cx < frame.cols; ++cx)
			for(int cy=0; cy < frame.rows; ++cy)
				
				for(int x = cx - half; x <= cx + half; ++x)
					for(int y = cy - half; y <= cy + half; ++y)
						if(inside(x, y, 0, 0, frame.cols, frame.rows))
							weights.at<float>(cv::Vec<int,4> {cy, cx, y -cy +half, x -cx +half}) = weight(frame(cy, cx), frame(y, x), this->gamma);
	}
	
	
	
	void PatchMatch::planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat1f &disp)
	{
		//cv::Mat1f raw_disp(planes.rows, planes.cols);
		
		#pragma omp parallel for
		for(int x=0; x < cols; ++x)
			for(int y=0; y < rows; ++y)
				disp(y, x) = disparity(x, y, planes(y, x));
			
		//cv::normalize(raw_disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}

	
	
	void PatchMatch::initialize_random_planes(Matrix2D<Plane> &planes, float max_d)
	{
		cv::RNG random_generator;
		const int RAND_HALF = RAND_MAX / 2;
		
		#pragma omp parallel for
		for(int y=0; y<rows; ++y)
		{
			for(int x=0; x<cols; ++x)
			{
				float z = random_generator.uniform(.0f, max_d); // random disparity
				cv::Vec3f point(x, y, z);
				
				float nx = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				float ny = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				float nz = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				cv::Vec3f normal(nx,ny,nz);
				cv::normalize(normal, normal);
				
				planes(y, x) = Plane(point, normal);
			}
		}
	}
	
	
	
	void PatchMatch::evaluate_planes_cost(int cpv)
	{
		#pragma omp parallel for
		for(int y=0; y<rows; ++y)
			for(int x=0; x<cols; ++x)
				costs[cpv](y, x) = plane_match_cost(planes[cpv](y,x), x, y, WINDOW_SIZE, cpv);
	}
	
	
	
	// search for better plane in the neighbourhood of a pixel
	// if iter is even then the function check the left and upper neighbours
	// if iter is odd then the function check the right and lower neighbours
	void PatchMatch::spatial_propagation(int x, int y, int cpv, int iter)
	{
		//std::cerr<<"START SPATIAL PROP\n";
		int rows = views[cpv].rows;
		int cols = views[cpv].cols;
		
		std::vector<std::pair<int, int>> offsets;
		
		if(iter%2 == 0)
		{
			offsets.push_back(std::make_pair(-1, 0));
			offsets.push_back(std::make_pair(0, -1));	
		}
		else
		{
			offsets.push_back(std::make_pair(+1, 0));
			offsets.push_back(std::make_pair(0, +1));	
		}
		
		int sign = (cpv == 0) ? -1 : 1;
		
		Plane& old_plane = planes[cpv](y, x);
		float& old_cost  = costs[cpv](y, x);
		
		for(auto it = offsets.begin(); it < offsets.end(); ++it)
		{
			std::pair<int, int> ofs = *it;
			
			int ny = y + ofs.first;
			int nx = x + ofs.second;
			
			if(!inside(nx, ny, 0, 0, cols, rows))
				continue;
			
			Plane p_neigb = planes[cpv](ny, nx);
			float new_cost = plane_match_cost(p_neigb, x, y, WINDOW_SIZE, cpv);
			
			if(new_cost < old_cost)
			{
				old_plane = p_neigb;
				old_cost = new_cost;
			}
		}
	}

		
	
	void PatchMatch::view_propagation(int x, int y, int cpv)
	{
		int sign = (cpv == 0) ? -1 : 1;	// -1 processing left, +1 processing right
		
		// current plane
		Plane view_plane = planes[cpv](y, x);
		
		// computing matching point in other view		
		// reparameterized corresopndent plane in other view
		int mx, my;
		Plane new_plane = view_plane.viewTransform(x, y, sign, mx, my);
		
		if(!inside(mx, my, 0, 0, views[0].cols, views[0].rows))
			return;
		
		// check if this reparameterized plane is better in the other view
		float& old_cost = costs[1-cpv](my, mx);
		float  new_cost = plane_match_cost(new_plane, mx, my, WINDOW_SIZE, 1-cpv);
		
		if(new_cost < old_cost)
		{
			planes[1-cpv](my, mx) = new_plane;
			old_cost = new_cost;
		}
	}

	
	
	void PatchMatch::plane_refinement(int x, int y, int cpv, float max_delta_z, float max_delta_n, float end_dz)
	{	
		int sign = (cpv == 0) ? -1 : 1;	// -1 processing left, +1 processing right
		
		float max_dz  = max_delta_z;
		float max_dn = max_delta_n;
		
		Plane& old_plane = planes[cpv](y, x);
		float& old_cost = costs[cpv](y, x);
		
		while(max_dz >= end_dz)
		{
			// Searching a random plane starting from the actual one
			std::random_device rd;
			std::mt19937 gen(rd());
			
			std::uniform_real_distribution<> rand_z(-max_dz, +max_dz);
			std::uniform_real_distribution<> rand_n(-max_dn, +max_dn);
			
			float z = old_plane[0] * x + old_plane[1] * y + old_plane[2];
			float delta_z = rand_z(gen);
			cv::Vec3f new_point(x, y, z + delta_z);
			
			cv::Vec3f n = old_plane.getNormal();
			cv::Vec3f delta_n(rand_n(gen), rand_n(gen), rand_n(gen));
			cv::Vec3f new_normal = n + delta_n;
			cv::normalize(new_normal, new_normal);
			
			// test the new plane
			Plane new_plane(new_point, new_normal);
			float new_cost = plane_match_cost(new_plane, x, y, WINDOW_SIZE, cpv);
			
			if(new_cost < old_cost)
			{
				old_plane = new_plane;
				old_cost = new_cost;
			}

			max_dz /= 2.0f;
			max_dn /= 2.0f;
		}
	}
	
	
	
	void PatchMatch::process_pixel(int x, int y, int cpv, int iter)
	{
		// spatial propagation
		spatial_propagation(x, y, cpv, iter);
		
		// plane refinement
		plane_refinement(x, y, cpv, MAX_DISPARITY/2, 1.0f, 0.1f);
		
		// view propagation
		view_propagation(x, y, cpv);
	}
	
	
	
	void PatchMatch::fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes, const cv::Mat1b &validity)
	{
		int x_lft = x - 1;
		int x_rgt = x + 1;
		
		while(!validity(y, x_lft) && x_lft >= 0)
		--x_lft;
		
		while(!validity(y, x_rgt) && x_lft < cols)
			++x_rgt;
		
		int best_plane_x = x;
		
		if(x_lft >= 0 && x_rgt < cols)
		{
			float disp_l = disparity(x, y, planes(y, x_lft));
			float disp_r = disparity(x, y, planes(y, x_rgt));
			best_plane_x = (disp_l < disp_r) ? x_lft : x_rgt;
		}
		else if(x_lft >= 0)
			best_plane_x = x_lft;
		else if(x_rgt < cols)
			best_plane_x = x_rgt;

		planes(y, x) = planes(y, best_plane_x);
	}
	
	
	
	void PatchMatch::weighted_median_filter(int cx, int cy, cv::Mat1f &disparity, const cv::Mat &weights, const cv::Mat1b &valid, int ws, bool use_invalid)
	{
		int half = ws / 2;
		float w_tot = 0;
		float w = 0;
		
		std::vector<std::pair<float, float>> disps_w;
		
		for(int x = cx-half; x <= cx + half; ++x)
			for(int y = cy-half; y <= cy + half; ++y)
				if(inside(x, y, 0, 0, cols, rows) && (use_invalid || valid(y, x)))
				{
					cv::Vec<int, 4> w_ids({cy, cx, y -cy +half, x -cx +half});
					
					w_tot += weights.at<float>(w_ids);
					disps_w.push_back(std::make_pair(weights.at<float>(w_ids), disparity(y,x)));
				}
		
		std::sort(disps_w.begin(), disps_w.end());
		
		float med_w = w_tot / 2.0f;
		
		for(auto dw=disps_w.begin(); dw < disps_w.end(); ++dw)
		{
			w += dw->first;
			
			if(w >= med_w)
			{
				if(dw == disps_w.begin())
				{
					disparity(cy, cx) = dw->second;
				}	
				else
				{
					disparity(cy, cx) = ((dw-1)->second + dw->second) / 2.0f;
				}
				//disparity(cy, cx) = dw->second;
			}
		}
	}
	
	
	void PatchMatch::operator()(const cv::Mat3b &img1, const cv::Mat3b &img2, int iterations, bool reverse)
	{
		this->set(img1, img2);
		this->process(iterations, reverse);
		this->postProcess();
	}
	
	
	
	void PatchMatch::set(const cv::Mat3b &img1, const cv::Mat3b &img2)
	{
		this->views[0] = img1;
		this->views[1] = img2;
		
		this->rows = img1.rows;
		this->cols = img1.cols;
		
		// pixels neighbours weights
		std::cerr<<"Precomputing pixels weight...\n";
		int wmat_sizes[] = {rows, cols, WINDOW_SIZE, WINDOW_SIZE};
		this->weigs[0] = cv::Mat(4, wmat_sizes, CV_32F);
		this->weigs[1] = cv::Mat(4, wmat_sizes, CV_32F);
		precompute_pixels_weights(img1, this->weigs[0], WINDOW_SIZE);
		precompute_pixels_weights(img2, this->weigs[1], WINDOW_SIZE);
		
		// greyscale images gradient
		std::cerr<<"Evaluating images gradient...\n";
		this->grads[0] = cv::Mat2f(rows, cols);
		this->grads[1] = cv::Mat2f(rows, cols);
		compute_greyscale_gradient(img1, this->grads[0]);
		compute_greyscale_gradient(img2, this->grads[1]);
		
		// pixels' planes random inizialization
		std::cerr<<"Precomputing random planes...\n";
		this->planes[0] = Matrix2D<Plane>(rows, cols);
		this->planes[1] = Matrix2D<Plane>(rows, cols);
		this->initialize_random_planes(this->planes[0], MAX_DISPARITY);
		this->initialize_random_planes(this->planes[1], MAX_DISPARITY);
		
		// initial planes costs evaluation
		std::cerr<<"Evaluating initial planes cost...\n";
		this->costs[0] = cv::Mat1f(rows, cols);
		this->costs[1] = cv::Mat1f(rows, cols);
		this->evaluate_planes_cost(0);
		this->evaluate_planes_cost(1);
		
		// left and right disparity maps
		this->disps[0] = cv::Mat1f(rows, cols);
		this->disps[1] = cv::Mat1f(rows, cols);
	}
	
	
	
	
	void PatchMatch::process(int iterations, bool reverse)
	{
		std::cerr <<"Processing left and right views...\n";
		for(int iter = 0 + reverse;  iter < iterations + reverse;  ++iter)
		{
			bool iter_type = (iter % 2 == 0);
			std::cerr<<"Iteration "<<iter-reverse+1 <<"/" <<iterations-reverse <<"\r";
			
			// PROCESS LEFT AND RIGHT VIEW IN SEQUENCE
			for(int work_view=0; work_view < 2; ++work_view)
			{
				if(iter_type)
				{
					for(int y=0; y < rows; ++y)
						for(int x=0; x < cols; ++x)
							process_pixel(x, y, work_view, iter);
				}
				else
				{
					for(int y = rows-1; y >= 0; --y)
						for(int x = cols-1; x >= 0; --x)
							process_pixel(x, y, work_view, iter);
				}
			}
		}
		std::cerr<<std::endl;
		
		this->planes_to_disparity(this->planes[0], this->disps[0]);
		this->planes_to_disparity(this->planes[1], this->disps[1]);
	}
	
	
	
	void PatchMatch::postProcess()
	{
		std::cerr<<"Executing post-processing...\n";
		
		// checking pixels-plane disparity validity
		cv::Mat1b lft_validity(rows, cols, (unsigned char)false);
		cv::Mat1b rgt_validity(rows, cols, (unsigned char)false);
		
		// cv::Mat1b ld(rows, cols);
		// cv::Mat1b rd(rows, cols);
		
		for(int y=0; y < rows; ++y)
		{
			for(int x=0; x < cols; ++x)
			{
				int x_rgt_match = std::max(0.f, std::min((float)cols, x - disps[0](y, x)));
				lft_validity(y, x) = (std::abs(disps[0](y, x) - disps[1](y, x_rgt_match)) <= 1);

				int x_lft_match = std::max(0.f, std::min((float)rows, x + disps[1](y, x)));
				rgt_validity(y, x) = (std::abs(disps[1](y, x) - disps[0](y, x_lft_match)) <= 1);
			}
		}
		
		// cv::imwrite("l_inv.png", 255*lft_validity);
		// cv::imwrite("r_inv.png", 255*rgt_validity);
		
		// fill-in holes related to invalid pixels
		#pragma omp parallel for
		for(int y=0; y < rows; y++)
		{
			for (int x=0; x < cols; x++)
			{
				if (!lft_validity(y, x))
					fill_invalid_pixels(y, x, planes[0], lft_validity);
				
				if (!rgt_validity(y, x))
					fill_invalid_pixels(y, x, planes[1], rgt_validity);
			}
		}
		
		this->planes_to_disparity(this->planes[0], this->disps[0]);
		this->planes_to_disparity(this->planes[1], this->disps[1]);
		
		// cv::normalize(disps[0], ld, 0, 255, cv::NORM_MINMAX);
		// cv::normalize(disps[1], rd, 0, 255, cv::NORM_MINMAX);
		// cv::imwrite("ld2.png", ld);
		// cv::imwrite("rd2.png", rd);
		
		// applying weighted median filter to left and right view respectively
		for(int x=0; x<cols; ++x)
		{
			for(int y=0; y<rows; ++y)
			{
				weighted_median_filter(x, y, disps[0], weigs[0], lft_validity, WINDOW_SIZE, false);
				weighted_median_filter(x, y, disps[1], weigs[1], rgt_validity, WINDOW_SIZE, false);
			}
		}
	}

}




