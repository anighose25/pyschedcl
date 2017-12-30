#define INVALID -2 

inline float sq(float r) {
	return r * r;
}

//removed M_data[4] and split it into parts
inline float3 Mat4TimeFloat3(float4 M_data_0,float4 M_data_1,float4 M_data_2,float4 M_data_3, float3 v) {
	return (float3)(
			dot((float3)(M_data_0.x,M_data_0.y,M_data_0.z), v)
					+ M_data_0.w,
			dot((float3)(M_data_1.x,M_data_1.y,M_data_1.z), v)
					+ M_data_1.w,
			dot((float3)(M_data_2.x,M_data_2.y,M_data_2.z), v)
					+M_data_2.w);
}

inline void setVolume(uint3 v_size, float3 v_dim, __global short2 *v_data, uint3 pos, float2 d) {
	v_data[pos.x + pos.y * v_size.x + pos.z * v_size.x * v_size.y] = (short2)(
			d.x * 32766.0f, d.y);
}

inline float3 posVolume(const uint3 v_size, const float3 v_dim, const __global short2 *v_data, const uint3 p) {
	return (float3)((p.x + 0.5f) * v_dim.x / v_size.x,
			(p.y + 0.5f) * v_dim.y / v_size.y,
			(p.z + 0.5f) * v_dim.z / v_size.z);
}

inline float2 getVolume(const uint3 v_size, const float3 v_dim, const __global short2* v_data, const uint3 pos) {
	const short2 d = v_data[pos.x + pos.y * v_size.x
			+ pos.z * v_size.x * v_size.y];
	return (float2)(d.x * 0.00003051944088f, d.y); //  / 32766.0f
}

inline float vs(const uint3 pos, const uint3 v_size, const float3 v_dim, const __global short2* v_data) {
	return v_data[pos.x + pos.y * v_size.x + pos.z * v_size.x * v_size.y].x;
}


inline float interp(const float3 pos, const uint3 v_size, const float3 v_dim, const __global short2 *v_data) {
	const float3 scaled_pos = (float3)((pos.x * v_size.x / v_dim.x) - 0.5f,
			(pos.y * v_size.y / v_dim.y) - 0.5f,
			(pos.z * v_size.z / v_dim.z) - 0.5f);

	float3 basef = (float3)(0);
	
	const int3 base = convert_int3(floor(scaled_pos));

	const float3 factor = (float3)(fract(scaled_pos, (float3 *) &basef));

	const int3 lower = max(base, (int3)(0));

	const int3 upper = min(base + (int3)(1), convert_int3(v_size) - (int3)(1));

	return (  (  ( vs  ( (uint3)(lower.x, lower.y, lower.z), v_size, v_dim, v_data) * (1 - factor.x)
			+ vs((uint3)(upper.x, lower.y, lower.z), v_size, v_dim, v_data) * factor.x)
			* (1 - factor.y)
			+ (vs((uint3)(lower.x, upper.y, lower.z), v_size, v_dim, v_data) * (1 - factor.x)
					+ vs((uint3)(upper.x, upper.y, lower.z), v_size, v_dim, v_data) * factor.x)
					* factor.y) * (1 - factor.z)
			+ ((vs((uint3)(lower.x, lower.y, upper.z), v_size, v_dim, v_data) * (1 - factor.x)
					+ vs((uint3)(upper.x, lower.y, upper.z), v_size, v_dim, v_data) * factor.x)
					* (1 - factor.y)
					+ (vs((uint3)(lower.x, upper.y, upper.z), v_size, v_dim, v_data)
							* (1 - factor.x)
							+ vs((uint3)(upper.x, upper.y, upper.z), v_size, v_dim, v_data)
									* factor.x) * factor.y) * factor.z)
			* 0.00003051944088f;
}

// Changed from Volume
inline float3 grad(float3 pos, const uint3 v_size, const float3 v_dim, const __global short2 *v_data) {
	const float3 scaled_pos = (float3)((pos.x * v_size.x / v_dim.x) - 0.5f,
			(pos.y * v_size.y / v_dim.y) - 0.5f,
			(pos.z * v_size.z / v_dim.z) - 0.5f);
	const int3 base = (int3)(floor(scaled_pos.x), floor(scaled_pos.y),
			floor(scaled_pos.z));
	const float3 basef = (float3)(0);
	const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
	const int3 lower_lower = max(base - (int3)(1), (int3)(0));
	const int3 lower_upper = max(base, (int3)(0));
	const int3 upper_lower = min(base + (int3)(1),
			convert_int3(v_size) - (int3)(1));
	const int3 upper_upper = min(base + (int3)(2),
			convert_int3(v_size) - (int3)(1));
	const int3 lower = lower_upper;
	const int3 upper = upper_lower;

	float3 gradient;

	gradient.x = (((vs((uint3)(upper_lower.x, lower.y, lower.z), v_size, v_dim, v_data)
			- vs((uint3)(lower_lower.x, lower.y, lower.z),v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper_upper.x, lower.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_upper.x, lower.y, lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(upper_lower.x, upper.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_lower.x, upper.y, lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, upper.y, lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_upper.x, upper.y, lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(upper_lower.x, lower.y, upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower_lower.x, lower.y, upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, lower.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_upper.x, lower.y, upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(upper_lower.x, upper.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower_lower.x, upper.y, upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper_upper.x, upper.y, upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(lower_upper.x, upper.y,
													upper.z),  v_size, v_dim, v_data)) * factor.x)
							* factor.y) * factor.z;

	gradient.y = (((vs((uint3)(lower.x, upper_lower.y, lower.z),  v_size, v_dim, v_data)
			- vs((uint3)(lower.x, lower_lower.y, lower.z),  v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, upper_lower.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(upper.x, lower_lower.y, lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper_upper.y, lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower_upper.y, lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_upper.y, lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower_upper.y, lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, upper_lower.y, upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower_lower.y, upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_lower.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower_lower.y, upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper_upper.y, upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower.x, lower_upper.y, upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper_upper.y, upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(upper.x, lower_upper.y,
													upper.z),  v_size, v_dim, v_data)) * factor.x)
							* factor.y) * factor.z;

	gradient.z = (((vs((uint3)(lower.x, lower.y, upper_lower.z),  v_size, v_dim, v_data)
			- vs((uint3)(lower.x, lower.y, lower_lower.z),  v_size, v_dim, v_data)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, lower.y, upper_lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(upper.x, lower.y, lower_lower.z),  v_size, v_dim, v_data))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper.y, upper_lower.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, upper.y, lower_lower.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper.y, upper_lower.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, upper.y, lower_lower.z),  v_size, v_dim, v_data))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, lower.y, upper_upper.z),  v_size, v_dim, v_data)
					- vs((uint3)(lower.x, lower.y, lower_upper.z),  v_size, v_dim, v_data))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, lower.y, upper_upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(upper.x, lower.y, lower_upper.z),  v_size, v_dim, v_data))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper.y, upper_upper.z),  v_size, v_dim, v_data)
							- vs((uint3)(lower.x, upper.y, lower_upper.z),  v_size, v_dim, v_data))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper.y, upper_upper.z),  v_size, v_dim, v_data)
									- vs(
											(uint3)(upper.x, upper.y,
													lower_upper.z),  v_size, v_dim, v_data))
									* factor.x) * factor.y) * factor.z;

	return gradient
			* (float3)(v_dim.x / v_size.x, v_dim.y / v_size.y,
					v_dim.z / v_size.z) * (0.5f * 0.00003051944088f);
}


// Changed from Matrix4
inline float3 get_translation(const float4 view_data_0,const float4 view_data_1,const float4 view_data_2,const float4 view_data_3) {
	return (float3)(view_data_0.w, view_data_1.w, view_data_2.w);
}


// Changed from Matrix4
inline float3 myrotate(const float4 M_data_0,const float4 M_data_1,const float4 M_data_2,const float4 M_data_3,const float3 v) {
	return (float3)(dot((float3)(M_data_0.x, M_data_0.y, M_data_0.z), v),
			dot((float3)(M_data_1.x, M_data_1.y, M_data_1.z), v),
			dot((float3)(M_data_2.x, M_data_2.y, M_data_2.z), v));
}


// Changed from Volume, Matrix4
float4 raycast(const uint3 v_size, const float3 v_dim, const __global short2 *v_data, const uint2 pos, const float4 view_data_0,
		const float4 view_data_1,const float4 view_data_2,const float4 view_data_3,const float nearPlane, const float farPlane,
		const float step,const float largestep) {

	const float3 origin = get_translation(view_data_0,view_data_1,view_data_2,view_data_3);
	const float3 direction = myrotate(view_data_0,view_data_1,view_data_2,view_data_3, (float3)(pos.x, pos.y, 1.f));

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = (float3)(1.0f) / direction;
	const float3 tbot = (float3) - 1 * invR * origin;
	const float3 ttop = invR * (v_dim - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fmin(ttop, tbot);
	const float3 tmax = fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmax(fmax(tmin.x, tmin.y), fmax(tmin.x, tmin.z));
	const float smallest_tmax = fmin(fmin(tmax.x, tmax.y),
			fmin(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar = fmin(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = interp(origin + direction * t, v_size, v_dim, v_data);
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
				f_tt = interp(origin + direction * t, v_size, v_dim, v_data);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return (float4)(origin + direction * t, t);
			}
		}
	}

	return (float4)(0);
}


/****************** KERNEL **************/


__kernel void initVolumeKernel(__global short2 * data) {

	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint z = get_global_id(2);
	uint3 size = (uint3) (get_global_size(0),get_global_size(1),get_global_size(2));
	float2 d = (float2) (1.0f,0.0f);

	data[x + y * size.x + z * size.x * size.y] = (short2) (d.x * 32766.0f, d.y);

}

__kernel void bilateralFilterKernel( __global float * out,
		const __global float * in,
		const __global float * gaussian,
		const float e_d,
		const int r ) {

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const uint2 size = (uint2) (get_global_size(0),get_global_size(1));

	const float center = in[pos.x + size.x * pos.y];

	if ( center == 0 ) {
		out[pos.x + size.x * pos.y] = 0;
		return;
	}

	float sum = 0.0f;
	float t = 0.0f;
	// FIXME : sum and t diverge too much from cpp version
	for(int i = -r; i <= r; ++i) {
		for(int j = -r; j <= r; ++j) {
			const uint2 curPos = (uint2)(clamp(pos.x + i, 0u, size.x-1), clamp(pos.y + j, 0u, size.y-1));
			const float curPix = in[curPos.x + curPos.y * size.x];
			if(curPix > 0) {
				const float mod = sq(curPix - center);
				const float factor = gaussian[i + r] * gaussian[j + r] * exp(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			} else {
				//std::cerr << "ERROR BILATERAL " <<pos.x+i<< " "<<pos.y+j<< " " <<curPix<<" \n";
			}
		}
	}
	out[pos.x + size.x * pos.y] = t / sum;

}

__kernel void depth2vertexKernel( __global float * vertex, // float3
		const uint2 vertexSize ,	//1
		const __global float * depth,
		const uint2 depthSize ,		//3
		const float4 invK_data_0,	//4
		const float4 invK_data_1,	//5
		const float4 invK_data_2,	//6
		const float4 invK_data_3	//7

		 ) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	float3 vert = (float3)(get_global_id(0),get_global_id(1),1.0f);

	if(pixel.x >= depthSize.x || pixel.y >= depthSize.y ) {
		return;
	}

	float3 res = (float3) (0);

	if(depth[pixel.x + depthSize.x * pixel.y] > 0) {
		res = depth[pixel.x + depthSize.x * pixel.y] * (myrotate(invK_data_0,invK_data_1,invK_data_2,invK_data_3, (float3)(pixel.x, pixel.y, 1.f)));
	}

	vstore3(res, pixel.x + vertexSize.x * pixel.y,vertex); 	// vertex[pixel] =

}

__kernel void halfSampleRobustImageKernel(__global float * out,
		__global const float * in,
		const uint2 inSize,
		const float e_d,
		const int r) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	uint2 outSize = inSize / 2;

	const uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x + centerPixel.y * inSize.x];
	for(int i = -r + 1; i <= r; ++i) {
		for(int j = -r + 1; j <= r; ++j) {
			int2 from = (int2)(clamp((int2)(centerPixel.x + j, centerPixel.y + i), (int2)(0), (int2)(inSize.x - 1, inSize.y - 1)));
			float current = in[from.x + from.y * inSize.x];
			if(fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel.x + pixel.y * outSize.x] = t / sum;

}

__kernel void integrateKernel (
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,
		__global const float * depth,
		const uint2 depthSize,

		float4 invTrack_data_0,
		float4 invTrack_data_1,
		float4 invTrack_data_2,
		float4 invTrack_data_3,

		float4 K_data_0,
		float4 K_data_1,
		float4 K_data_2,
		float4 K_data_3,

		const float mu,
		const float maxweight ,
		const float3 delta ,
		const float3 cameraDelta
) {

	// Volume vol; vol.data = v_data; vol.size = v_size; vol.dim = v_dim;
	//Removed this and used v_data, v_size,v_dim wherever vol is present

	uint3 pix = (uint3) (get_global_id(0),get_global_id(1),0);
	const int sizex = get_global_size(0);

	//posVolume uses size and dim of Volume only
	float3 pos = Mat4TimeFloat3 (invTrack_data_0 ,invTrack_data_1,invTrack_data_2,invTrack_data_3, posVolume(v_size, v_dim, v_data ,pix));
	float3 cameraX = Mat4TimeFloat3 ( K_data_0, K_data_1, K_data_2, K_data_3, pos);

	for(pix.z = 0; pix.z < v_size.z; ++pix.z, pos += delta, cameraX += cameraDelta) {
		if(pos.z < 0.0001f) // some near plane constraint
		continue;
		const float2 pixel = (float2) (cameraX.x/cameraX.z + 0.5f, cameraX.y/cameraX.z + 0.5f);

		if(pixel.x < 0 || pixel.x > depthSize.x-1 || pixel.y < 0 || pixel.y > depthSize.y-1)
		continue;
		const uint2 px = (uint2)(pixel.x, pixel.y);
		float depthpx = depth[px.x + depthSize.x * px.y];

		if(depthpx == 0) continue;
		const float diff = ((depthpx) - cameraX.z) * sqrt(1+sq(pos.x/pos.z) + sq(pos.y/pos.z));

		if(diff > -mu) {
			const float sdf = fmin(1.f, diff/mu);
			float2 data = getVolume(v_size, v_dim, v_data, pix);
			data.x = clamp((data.y*data.x + sdf)/(data.y + 1), -1.f, 1.f);
			data.y = fmin(data.y+1, maxweight);
			setVolume(v_size, v_dim, v_data, pix, data);
		}

	}

}


__kernel void mm2metersKernel(
		__global float * depth,
		const uint2 depthSize ,
		const __global ushort * in ,
		const uint2 inSize ,
		const int ratio ) {
	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	depth[pixel.x + depthSize.x * pixel.y] = in[pixel.x * ratio + inSize.x * pixel.y * ratio] / 1000.0f;
}


__kernel void raycastKernel( __global float * pos3D,  //float3
		__global float * normal,//float3
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,

		const float4 view_data_0,
		const float4 view_data_1,
		const float4 view_data_2,
		const float4 view_data_3,

		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep ) {

	//const Volume volume = {v_size, v_dim,v_data};				*Removed this and used its members every where

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const int sizex = get_global_size(0);

	const float4 hit = raycast( v_size,v_dim,v_data, pos, view_data_0,view_data_1,view_data_2,view_data_3, nearPlane, farPlane, step, largestep );
	const float3 test = as_float3(hit);

	if(hit.w > 0.0f ) {
		vstore3(test,pos.x + sizex * pos.y,pos3D);
		float3 surfNorm = grad(test,v_size,v_dim,v_data);
		if(length(surfNorm) == 0) {
			//float3 n =  (INVALID,0,0);//vload3(pos.x + sizex * pos.y,normal);
			//n.x=INVALID;
			vstore3((float3)(INVALID,INVALID,INVALID),pos.x + sizex * pos.y,normal);
		} else {
			vstore3(normalize(surfNorm),pos.x + sizex * pos.y,normal);
		}
	} else {
		vstore3((float3)(0),pos.x + sizex * pos.y,pos3D);
		vstore3((float3)(INVALID, INVALID, INVALID),pos.x + sizex * pos.y,normal);
	}
}



__kernel void reduceKernel (
		__global float * out,
		__global const int* J_result,
		__global const float* J_error,
		__global const float* J_J,
		const uint2 JSize,
		const uint2 size,
		__local float * S
) {

/* The float array must be filled in this form such as  [0,1,2,3,4,5] [0,1,2,3,4,5] ....
	where each box is an array index which is having another array of size 6 starting from its point */


	uint blockIdx = get_group_id(0);
	uint blockDim = get_local_size(0);
	uint threadIdx = get_local_id(0);
	uint gridDim = get_num_groups(0);

	const uint sline = threadIdx;

	float sums[32];
	float * jtj = sums + 7;
	float * info = sums + 28;

	for(uint i = 0; i < 32; ++i)
	sums[i] = 0.0f;

	for(uint y = blockIdx; y < size.y; y += gridDim) {
		for(uint x = sline; x < size.x; x += blockDim ) {


			int temp=x+y*JSize.x;	//use this variable


			//const TrackData row = J[x + y * JSize.x];

			if(J_result[temp]< 1) {
				info[1] +=  J_result[temp]==-4 ? 1 : 0;
				info[2] +=  J_result[temp] == -5 ? 1 : 0;
				info[3] +=  J_result[temp] > -4 ? 1 : 0;
				continue;
			}

			// Error part
			sums[0] +=  J_error[temp] *  J_error[temp];

			// JTe part
			for(int i = 0; i < 6; ++i)
			sums[i+1] +=  J_error[temp] * J_J[6*temp+i];	//padding done

			jtj[0] += J_J[6*temp] * J_J[6*temp];
			jtj[1] += J_J[6*temp] * J_J[6*temp+1];
			jtj[2] += J_J[6*temp] * J_J[6*temp+2];
			jtj[3] += J_J[6*temp] * J_J[6*temp+3];
			jtj[4] += J_J[6*temp] * J_J[6*temp+4];
			jtj[5] += J_J[6*temp] * J_J[6*temp+5];

			jtj[6] += J_J[6*temp+1] * J_J[6*temp+1];
			jtj[7] += J_J[6*temp+1] * J_J[6*temp+2];
			jtj[8] += J_J[6*temp+1] * J_J[6*temp+3];
			jtj[9] += J_J[6*temp+1] * J_J[6*temp+4];
			jtj[10] +=J_J[6*temp+1] * J_J[6*temp+5];

			jtj[11] += J_J[6*temp+2] * J_J[6*temp+2];
			jtj[12] += J_J[6*temp+2] * J_J[6*temp+3];
			jtj[13] += J_J[6*temp+2] * J_J[6*temp+4];
			jtj[14] += J_J[6*temp+2] * J_J[6*temp+5];

			jtj[15] += J_J[6*temp+3] * J_J[6*temp+3];
			jtj[16] += J_J[6*temp+3] * J_J[6*temp+4];
			jtj[17] += J_J[6*temp+3] * J_J[6*temp+5];

			jtj[18] += J_J[6*temp+4] * J_J[6*temp+4];
			jtj[19] += J_J[6*temp+4] * J_J[6*temp+5];

			jtj[20] += J_J[6*temp+5] * J_J[6*temp+5];
			// extra info here
			info[0] += 1;

		}

	}

	for(int i = 0; i < 32; ++i) // copy over to shared memory
	S[sline * 32 + i] = sums[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(sline < 32) { // sum up columns and copy to global memory in the final 32 threads
		for(unsigned i = 1; i < blockDim; ++i)
		S[sline] += S[i * 32 + sline];
		out[sline+blockIdx*32] = S[sline];
	}
}


__kernel void renderDepthKernel( __global uchar4 * out,
		__global float * depth,
		const float nearPlane,
		const float farPlane ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);
	float d= depth[posx + sizex * posy];
	if(d < nearPlane)
	vstore4((uchar4)(255, 255, 255, 0), posx + sizex * posy, (__global uchar*)out); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
	else {
		if(d > farPlane)
		vstore4((uchar4)(0, 0, 0, 0), posx + sizex * posy, (__global uchar*)out);
		else {
			float h =(d - nearPlane) / (farPlane - nearPlane);
			h *= 6.0f;
			const int sextant = (int)h;
			const float fract = h - sextant;
			const float mid1 = 0.25f + (0.5f*fract);
			const float mid2 = 0.75f - (0.5f*fract);
			switch (sextant)
			{
				case 0: vstore4((uchar4)(191, 255*mid1, 64, 0), posx + sizex * posy, (__global uchar*)out); break;
				case 1: vstore4((uchar4)(255*mid2, 191, 64, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 2: vstore4((uchar4)(64, 191, 255*mid1, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 3: vstore4((uchar4)(64, 255*mid2, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 4: vstore4((uchar4)(255*mid1, 64, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 5: vstore4((uchar4)(191, 64, 255*mid2, 0),posx + sizex * posy ,(__global uchar*)out); break;
			}
		}
	}
}



__kernel void renderNormalKernel( const __global uchar * in,
					__global float * out )
{

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	float3 n;
	const uchar3 i = vload3(posx + sizex * posy,in);

	n.x = i.x;
	n.y = i.y;
	n.z = i.z;

	if(n.x == -2) {
		vstore3((float3) (0,0,0),posx + sizex * posy,out);
	} else {
		n = normalize(n);
		vstore3((float3) (n.x*128 + 128,n.y*128+128, n.z*128+128), posx + sizex * posy,out);
	}


}



//changed track data * pointer
__kernel void renderTrackKernel( __global uchar4 * out,
		__global const int * data_result ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	switch(data_result[posx + sizex * posy]) {
		// The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
		case  1: vstore4((uchar4)(128, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); break; // ok	 GREY
		case -1: vstore4((uchar4)(000, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no input BLACK
		case -2: vstore4((uchar4)(255, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // not in image RED
		case -3: vstore4((uchar4)(000, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no correspondence GREEN
		case -4: vstore4((uchar4)(000, 000, 255, 0), posx + sizex * posy, (__global uchar*)out); break; // too far away BLUE
		case -5: vstore4((uchar4)(255, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // wrong normal YELLOW
		default: vstore4((uchar4)(255, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); return;
	}
}



__kernel void renderVolumeKernel( __global uchar * render,	//uchar*
		__global short2 * v_data,	//2-D
		const uint3 v_size,
		const float3 v_dim,
		const float4 view_data_0,
		const float4 view_data_1,
		const float4 view_data_2,
		const float4 view_data_3,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep,
		const float3 light,
		const float3 ambient) {


	//const Volume v = {v_size, v_dim,v_data};						Removed this and

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const int sizex = get_global_size(0);

	float4 hit = raycast(v_size, v_dim, v_data, pos, view_data_0,view_data_1,view_data_2,view_data_3, nearPlane, farPlane,step, largestep);

	if(hit.w > 0) {
		const float3 test = as_float3(hit);
		float3 surfNorm = grad(test,v_size,v_dim,v_data);
		if(length(surfNorm) > 0) {
			const float3 diff = normalize(light - test);
			const float dir = fmax(dot(normalize(surfNorm), diff), 0.f);
			const float3 col = clamp((float3)(dir) + ambient, 0.f, 1.f) * (float3) 255;
			vstore4((uchar4)(col.x, col.y, col.z, 0), pos.x + sizex * pos.y, render); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
		} else {
			vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
		}
	} else {
		vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
	}

}


__kernel void trackKernel (

		__global int* output_result,	//0
		__global float* output_error,	//1
		__global float* output_J,  //global array of size 6*dataset 2

		const uint2 outputSize,	//3
		__global const float * inVertex,// float3 4
		const uint2 inVertexSize,	//5
		__global const float * inNormal,// float3	//6
		const uint2 inNormalSize,	//7
		__global const float * refVertex,// float3	//8
		const uint2 refVertexSize,	//9
		__global const float * refNormal,// float3 10
		const uint2 refNormalSize,	//11

		const float4 Ttrack_data_0,	//12
		const float4 Ttrack_data_1,	//13
		const float4 Ttrack_data_2,	//14
		const float4 Ttrack_data_3,	//15

		const float4 view_data_0,	//16
		const float4 view_data_1,	//17
		const float4 view_data_2,	//18
		const float4 view_data_3,	//19

		const float dist_threshold,	//20
		const float normal_threshold	//21

						)

{

	const uint2 pixel = (uint2)(get_global_id(0),get_global_id(1));

	if(pixel.x >= inVertexSize.x || pixel.y >= inVertexSize.y ) {return;}

	float3 inNormalPixel = vload3(pixel.x + inNormalSize.x * pixel.y,inNormal);

	if(inNormalPixel.x == INVALID ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -1;
		return;
	}

	float3 inVertexPixel = vload3(pixel.x + inVertexSize.x * pixel.y,inVertex);
	const float3 projectedVertex = Mat4TimeFloat3 (Ttrack_data_0 ,Ttrack_data_1, Ttrack_data_2,Ttrack_data_3, inVertexPixel);
	const float3 projectedPos = Mat4TimeFloat3 ( view_data_0 , view_data_1 , view_data_2 , view_data_3 , projectedVertex);
	const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

	if(projPixel.x < 0 || projPixel.x > refVertexSize.x-1 || projPixel.y < 0 || projPixel.y > refVertexSize.y-1 ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -2;
		return;
	}

	const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
	const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

	if(referenceNormal.x == INVALID) {
		output_result[pixel.x + outputSize.x * pixel.y] = -3;
		return;
	}

	const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
	const float3 projectedNormal = myrotate(Ttrack_data_0,Ttrack_data_1,Ttrack_data_2,Ttrack_data_3, inNormalPixel);

	if(length(diff) > dist_threshold ) {
		output_result[pixel.x + outputSize.x * pixel.y] = -4;
		return;
	}
	if(dot(projectedNormal, referenceNormal) < normal_threshold) {
		output_result[pixel.x + outputSize.x * pixel.y] = -5;
		return;
	}

	output_result[pixel.x + outputSize.x * pixel.y] = 1;
	output_error[pixel.x + outputSize.x * pixel.y]  = dot(referenceNormal, diff);

	//added this here

	long d = pixel.x +outputSize.x*pixel.y;
	long f= (d-1)>=0 ? 6*(d-1):0;

	__private float temp[6]={0};
	temp[0]=output_J[f];
	temp[1]=output_J[f+1];
	temp[2]=output_J[f+2];
	temp[3]=output_J[f+3];
	temp[4]=output_J[f+4];
	temp[5]=output_J[f+5];

	vstore3( referenceNormal,0,temp );

	output_J[f]=temp[0];
	output_J[f+1]=temp[1];
	output_J[f+2]=temp[2];
	output_J[f+3]=temp[3];
	output_J[f+4]=temp[4];
	output_J[f+5]=temp[5];


	vstore3( cross(projectedVertex, referenceNormal),1,temp);

	output_J[f]=temp[0];
	output_J[f+1]=temp[1];
	output_J[f+2]=temp[2];
	output_J[f+3]=temp[3];
	output_J[f+4]=temp[4];
	output_J[f+5]=temp[5];


}



__kernel void vertex2normalKernel( __global float * normal,    // float3
		const uint2 normalSize,
		const __global float * vertex ,
		const uint2 vertexSize ) {  // float3

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));

	if(pixel.x >= vertexSize.x || pixel.y >= vertexSize.y )
	return;

	//const float3 left = vertex[(uint2)(max(int(pixel.x)-1,0), pixel.y)];
	//const float3 right = vertex[(uint2)(min(pixel.x+1,vertex.size.x-1), pixel.y)];
	//const float3 up = vertex[(uint2)(pixel.x, max(int(pixel.y)-1,0))];
	//const float3 down = vertex[(uint2)(pixel.x, min(pixel.y+1,vertex.size.y-1))];

	uint2 vleft = (uint2)(max((int)(pixel.x)-1,0), pixel.y);
	uint2 vright = (uint2)(min(pixel.x+1,vertexSize.x-1), pixel.y);
	uint2 vup = (uint2)(pixel.x, max((int)(pixel.y)-1,0));
	uint2 vdown = (uint2)(pixel.x, min(pixel.y+1,vertexSize.y-1));

	const float3 left = vload3(vleft.x + vertexSize.x * vleft.y,vertex);
	const float3 right = vload3(vright.x + vertexSize.x * vright.y,vertex);
	const float3 up = vload3(vup.x + vertexSize.x * vup.y,vertex);
	const float3 down = vload3(vdown.x + vertexSize.x * vdown.y,vertex);
	/*
	 unsigned long int val =  0 ;
	 val = max(((int) pixel.x)-1,0) + vertexSize.x * pixel.y;
	 const float3 left   = vload3(   val,vertex);

	 val =  min(pixel.x+1,vertexSize.x-1)                  + vertexSize.x *     pixel.y;
	 const float3 right  = vload3(    val     ,vertex);
	 val =   pixel.x                        + vertexSize.x *     max(((int) pixel.y)-1,0)  ;
	 const float3 up     = vload3(  val ,vertex);
	 val =  pixel.x                       + vertexSize.x *   min(pixel.y+1,vertexSize.y-1)   ;
	 const float3 down   = vload3(  val   ,vertex);
	 */
	if(left.z == 0 || right.z == 0|| up.z ==0 || down.z == 0) {
		//float3 n = vload3(pixel.x + normalSize.x * pixel.y,normal);
		//n.x=INVALID;
		vstore3((float3)(INVALID,INVALID,INVALID),pixel.x + normalSize.x * pixel.y,normal);
		return;
	}
	const float3 dxv = right - left;
	const float3 dyv = down - up;
	vstore3((float3) normalize(cross(dyv, dxv)), pixel.x + pixel.y * normalSize.x, normal );

}




