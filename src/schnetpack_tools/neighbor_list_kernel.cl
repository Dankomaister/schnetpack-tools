
__kernel void neighbor_list(__global double3 *scaled_positions, __global short3 *offsets, __global int *neighborhood_idx,
	uint n_atoms, uint max_nbh, double cutoff,
	double c11, double c12, double c13,
	double c21, double c22, double c23,
	double c31, double c32, double c33)
{
	uint i = get_global_id(0);
	uint k = max_nbh*i;

	double R_sq = 0.0;
	double cutoff_sq = cutoff*cutoff;

	double3 local_position = scaled_positions[i];
	double3 distance = (double3)(0.0);
	double3 offset = (double3)(0.0);

	double3 a = (double3)(c11, c12, c13);
	double3 b = (double3)(c21, c22, c23);
	double3 c = (double3)(c31, c32, c33);

	double3 denominator = dot(a, cross(b, c));

	int n1 = round(cutoff*length(cross(b, c)/denominator));
	int n2 = round(cutoff*length(cross(c, a)/denominator));
	int n3 = round(cutoff*length(cross(a, b)/denominator));

	for(int s1 = -n1; s1 < n1+1; s1++)
	{
	for(int s2 = -n2; s2 < n2+1; s2++)
	{
	for(int s3 = -n3; s3 < n3+1; s3++)
	{
		double3 shift = (double3)(s1, s2, s3);

		for (uint j = 0u; j < n_atoms; j++)
		{
			if (i == j && s1 == 0 && s2 == 0 && s3 == 0) continue;

			distance = scaled_positions[j] - local_position;
			offset   = -round(distance) + shift;
			distance = distance + offset;

			distance = distance.x*a + distance.y*b + distance.z*c;
			R_sq = distance.x*distance.x + distance.y*distance.y + distance.z*distance.z;

			if (R_sq < cutoff_sq)
			{
				neighborhood_idx[k] = j;
				offsets[k] = convert_short3(offset);
				k++;
			}
		}
	}
	}
	}

	if (k - max_nbh*i > max_nbh)
	{
		printf("\nWARNING! found %i neighbours for atom %i which larger than max_nbh=%i.\n         try and increase number_density", k - max_nbh*i, i, max_nbh);
	}
}
