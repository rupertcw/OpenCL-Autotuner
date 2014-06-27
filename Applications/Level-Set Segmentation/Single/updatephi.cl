#define ALPHA		 0.007f
#define DT			 0.2f

#define max(x,y)    ((x>y) ? x : y )
#define min(x,y)    ((x<y) ? x : y )

// CPU 1 ITER: 0.00332808
// GPU 1 ITER: 0.0208189
// Tiles + Operations
__kernel void update_phi(global float *d_phi, global float *d_phi1, global float *d_D, int imageW, int imageH, int tsx, int tsy)
{
    int c = get_global_id(0) * tsx;
	int r = get_global_id(1) * tsy;
    
    for (int py = r; py < r+tsy; py++) {
        for (int px = c; px < c+tsx; px++) {
            
            int ind= py*imageW + px;
            
            float dx, dxplus, dxminus, dxplusy, dxminusy;
            float dy, dyplus, dyminus, dyplusx, dyminusx;
            float gradphimax, gradphimin, nplusx, nplusy, nminusx, nminusy, curvature;
            float F, gradphi;
            
            if (c == 0 || c == imageW-1)
                dx = 0;
            else
                dx = native_divide((d_phi1[ind + 1] - d_phi1[ind - 1]), 2);

            if (c == imageW-1)
                dxplus = 0;
            else
                dxplus = (d_phi1[ind + 1] - d_phi1[ind]);
            
            if (c == 0)
                dxminus = 0;
            else
                dxminus = (d_phi1[ind] - d_phi1[ind - 1]);

            if (r == 0 || c == 0 || c == imageW-1)
                dxplusy = 0;
            else
                dxplusy = native_divide((d_phi1[ind - imageW + 1] - d_phi1[ind - imageW - 1]), 2);

            if (r == imageH-1 || c == 0 || c == imageW-1)
                dxminusy = 0;
            else
                dxminusy = native_divide((d_phi1[ind + imageW + 1] - d_phi1[ind + imageW - 1]), 2);

            if (r == 0 || r == imageH-1)
                dy = 0;
            else
                dy = native_divide((d_phi1[ind - imageW] - d_phi1[ind + imageW]), 2);

            if (r == 0)
                dyplus = 0;
            else
                dyplus = (d_phi1[ind - imageW] - d_phi1[ind]);

            if (r == imageH-1)
                dyminus = 0;
            else
                dyminus = (d_phi1[ind] - d_phi1[ind + imageW]);

            if (r == 0 || c == imageW-1 || r == imageH-1)
                dyplusx = 0;
            else
                dyplusx = native_divide((d_phi1[ind - imageW + 1] - d_phi1[ind + imageW + 1]), 2);

            if (r == 0 || c == 0 || r == imageH-1)
                dyminusx = 0;
            else
                dyminusx = native_divide((d_phi1[ind - imageW - 1] - d_phi1[ind + imageW - 1]), 2);

            float eq1 = (native_sqrt(mad(max(dxplus, 0), max(dxplus, 0), (max(-dxminus, 0)*max(-dxminus, 0)))));
            float eq2 = (native_sqrt(mad(max(dxplus, 0), max(dxplus, 0), (max(-dxminus, 0)*max(-dxminus, 0)))));
            float eq3 = (native_sqrt(mad(max(dyplus, 0), max(dyplus, 0), (max(-dyminus, 0)*max(-dyminus, 0)))));
            float eq4 = (native_sqrt(mad(max(dyplus, 0), max(dyplus, 0), (max(-dyminus, 0)*max(-dyminus, 0)))));

            gradphimax = (native_sqrt(mad(eq1, eq2, (eq3*eq4))));

            eq1 = (native_sqrt(mad(min(dxplus, 0), min(dxplus, 0), (min(-dxminus, 0)*min(-dxminus, 0)))));
            eq2 = (native_sqrt(mad(min(dxplus, 0), min(dxplus, 0), (min(-dxminus, 0)*min(-dxminus, 0)))));
            eq3 = (native_sqrt(mad(min(dyplus, 0), min(dyplus, 0), (min(-dyminus, 0)*min(-dyminus, 0)))));
            eq4 = (native_sqrt(mad(min(dyplus, 0), min(dyplus, 0), (min(-dyminus, 0)*min(-dyminus, 0)))));

            gradphimin = (native_sqrt(mad(eq1, eq2, (eq3*eq4))));
            
            nplusx = native_divide(dxplus, native_sqrt(1.192092896e-07F + mad(dxplus, dxplus, ((dyplusx + dy)*(dyplusx + dy)*0.25f))));
            nplusy = native_divide(dyplus, native_sqrt(1.192092896e-07F + mad(dyplus, dyplus, ((dxplusy + dx)*(dxplusy + dx)*0.25f))));
            nminusx = native_divide(dxminus, native_sqrt(1.192092896e-07F + mad(dxminus, dxminus, ((dyminusx + dy)*(dyminusx + dy)*0.25f))));
            nminusy = native_divide(dyminus, native_sqrt(1.192092896e-07F + mad(dyminus, dyminus, ((dxminusy + dx)*(dxminusy + dx)*0.25f))));
            curvature = native_divide(((nplusx - nminusx) + (nplusy - nminusy)), 2);
            
            F = mad(-ALPHA, d_D[ind], ((1 - ALPHA)*curvature));
            
            if (F > 0)
                gradphi = gradphimax;
            else
                gradphi = gradphimin;
            
            d_phi[ind] = d_phi1[ind] + (DT*F*gradphi);
        }
    }
}
