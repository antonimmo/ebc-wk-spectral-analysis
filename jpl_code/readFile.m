close all; clear; clc;


ncid = netcdf.open('Spectrum_778_at_LON_-59.0_LAT_31.462.nc');
[ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid);

for v=1:nvars
    varName{v} = netcdf.inqVar(ncid,v-1);
    var{v} = netcdf.getVar(ncid,v-1);
end

idx = find(var{3} <= 0);
var{3}(idx) = 1e-15;
figure
contourf(log(var{1}),log(var{2}),log(var{3}),100, 'LineStyle', 'none');
colorbar
caxis([-12 -10])

netcdf.close(ncid)
