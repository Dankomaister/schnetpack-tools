
__kernel void neighbor_list(__global double3 *scaled_positions, __global short3 *offset, __global int *neighborhood_idx,
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
	double3 ds = (double3)(0.0);
	double3 o  = (double3)(0.0);

	double3 a = (double3)(c11, c12, c13);
	double3 b = (double3)(c21, c22, c23);
	double3 c = (double3)(c31, c32, c33);

	double3 denominator = dot(a, cross(b, c));
	int n1 = round(cutoff*length(cross(b, c)/denominator));
	int n2 = round(cutoff*length(cross(c, a)/denominator));
	int n3 = round(cutoff*length(cross(a, b)/denominator));

	for(int r1 = -n1; r1 < n1+1; r1++)
	{
	for(int r2 = -n2; r2 < n2+1; r2++)
	{
	for(int r3 = -n3; r3 < n3+1; r3++)
	{
		double3 shift = (double3)(r1, r2, r3);

		for (uint j = 0u; j < n_atoms; j++)
		{
			if (i == j && r1 == 0 && r2 == 0 && r3 == 0) continue;

			ds = scaled_positions[j] - local_position;
			o  = -round(ds) + shift;
			ds = ds + o;

			ds = ds.x*a + ds.y*b + ds.z*c;
			R_sq = ds.x*ds.x + ds.y*ds.y + ds.z*ds.z;

			if (R_sq < cutoff_sq)
			{
				neighborhood_idx[k] = j;
				offset[k] = (double3)(o);
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
