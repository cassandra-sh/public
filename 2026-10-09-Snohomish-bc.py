import xarray as xr
import numpy as np
import os
import gc
import random
import utm
import glob
import geopandas as gpd
from scipy.interpolate import CubicSpline


def load_discharge(discharge_file, remap_file):
    """
    Load and remap the WFLOW discharge, to prepare it for SFINCS inputs
    """
    # Load discharge dataset
    discharge = xr.open_dataset(discharge_file)

    # Get discharge output points
    lat, lon = discharge.lat.values, discharge.lon.values
    points = gpd.points_from_xy(lon, lat, crs="EPSG:4326")
    # Convert to local CRS
    gdf = gpd.GeoDataFrame(geometry=points)
    gdf = gdf.to_crs('EPSG:32610')
    # Get the list of x, y coordinates as 1d arrays
    px = np.array([pt.xy[0][0] for pt in points])
    py = np.array([pt.xy[1][0] for pt in points])

    # Load discharge remap
    mapping = gpd.read_file(remap_file)

    # Pick the closest discharge point to each remapped location. Locations are the same as before but indexing has changed. 
    # find closest points (x0) and remapped location (x1) for each desired discharge point
    discharge_x, discharge_y = [], []
    discharge_indices = []
    # For each remapping linestring
    for s in mapping.geometry:
        # First point is the index to the discharge point
        x0,y0 = s.xy[0][0], s.xy[1][0]
        # Find the closest discharge point
        i = np.argmin(np.sqrt((px - x0)**2 + (py-y0)**2))
        discharge_indices.append(i)
        # Store the remapped location 
        discharge_x.append(s.xy[0][1])
        discharge_y.append(s.xy[1][1])
    discharge_indices = np.array(discharge_indices)
    discharge_x = np.array(discharge_x)
    discharge_y = np.array(discharge_y)
    xy = np.column_stack([discharge_x, discharge_y])

    # Load the discharge data
    t = discharge.time.values
    Qdiff = np.nanmean(discharge.Q_CmipDiff[:,:,discharge_indices].values, axis=0)
    Q = discharge.Q[:,discharge_indices].values
    return t, Q, Qdiff, xy

def xy_to_file(xy, path):
    np.savetxt(path, xy, fmt="%.3f")

def xy_from_waterlevel(waterlevel_path):
    waterlevel = xr.open_dataset(waterlevel_path)
    lat, lon = waterlevel.lat.values, waterlevel.lon.values
    x, y, _, __ = utm.from_latlon(lat, lon)
    return np.column_stack([x, y])

def make_inpfile(template_file, output_path, t0, tf, dtmapout='2592000'):
    """
    Generate sfincs.inp file

    @params
        t0 - start time [datetime64]. Will also be used as reference time. 
        tf - end time [datetime64]
        template_file - path to template file
        output_path - path to save new file
        dtmapount - string of seconds between map file outputs. default = 1 year to reduce filesize. 
    If you want the full time series, set to '1800' -- but be warned, file sizes will be big! 
    You can get full timseries using .his files (I am doing this here), in only the points you want 
    and/or a more sparse grid
    """
    input_template_file = open(template_file, 'r')
    input_template = input_template_file.read()

    tstart = str(t0)[:10].replace('-', '') + ' ' + str(t0)[11:].replace(':', '')
    tstop  = str(tf)[:10].replace('-', '') + ' ' + str(tf)[11:].replace(':', '')

    input_file_content = input_template.format(
        TSTART=tstart, 
        TSTOP=tstop, 
        TREF=tstart,
        DTMAPOUT=dtmapout
    )

    # Write the output
    input_file = open(output_path, "w")
    input_file.write(input_file_content)
    input_file.close()

def make_disfile(output_path, t0, tf, qt, Q):
    """
    Make sfincs.dis file with WFLOW discharge

    @params
        t0 - start time [datetime64]
        tf - end time [datetime64]
        qt - timestamps for Q
        Q  - Q itself. Shape = (time, stations)
        output_path - "/path/to/sfincs.bnd"
    """
    tok = np.logical_and(qt >= t0, qt <= tf)
    time_seconds = (qt[tok] - t0) / np.timedelta64(1, 's')
    out = np.column_stack([time_seconds, Q[tok]])
    fmt = ["%.1f"] + ["%.3f"] * Q[tok].shape[1]
    np.savetxt(output_path, out, fmt=fmt)

def nan_fix(x, xdiff, t, dx=0.1):
    """
    DFM Data have occasional NANs that need to be intelligently fixed with a 
    month-wise interpolation. 
    Apply month-by-month fix to xdiff data, assuming a stable xdiff(x, month)
    """
    xrange = np.arange(np.nanmin(x) + dx/2, np.nanmax(x) - dx/2, dx)
    month = t.astype('datetime64[M]').astype(int) % 12 + 1
    fixed = np.copy(xdiff)
    for m in range(1, 13):
        m_ok = month == m
        need_fix = m_ok & np.isnan(xdiff) & ~np.isnan(x)
        have_data = m_ok & ~np.isnan(xdiff) & ~np.isnan(x)
        if need_fix.sum() == 0 or have_data.sum() == 0:
            continue
        xg = x[have_data]
        dg = xdiff[have_data]
        # Get xdiff as function of x
        diff_func = np.full(xrange.shape, np.nan)
        for i, xm in enumerate(xrange):
            x_ok = (xg > xm - dx/2) & (xg < xm + dx/2)
            if x_ok.any():
                diff_func[i] = dg[x_ok].mean()
        # Drop empty bins
        dok = ~np.isnan(diff_func)
        if dok.sum() < 2:
            continue
        fixed[need_fix] = CubicSpline(xrange[dok], diff_func[dok], extrapolate=True)(x[need_fix])
    return fixed

def nan_fix_wrapper(quant, quant_diff, quant_time):
    """
    Handle nan fixing a quantity in shape (time, station), and skip if there are no nans
    """
    if np.sum(np.isnan(quant_diff)) > 0:
        return np.array([nan_fix(q, qd, quant_time) for q, qd in zip(quant.T, quant_diff.T)]).T
    else:
        return quant_diff

def nan_interp(quant, time):
    """
    There will also be instances of single nans in the dfm file. Interpolate over them.
    """
    quant_fixed = []
    for q in quant.T:
        if np.nansum(np.isnan(q)) == 0:
            quant_fixed.append(q)
        else:
            q = np.copy(q)
            q[np.isnan(q)] = np.interp(
                time[np.isnan(e)].astype('datetime64[ns]').astype(float), 
                time[~np.isnan(e)].astype('datetime64[ns]').astype(float), 
                q[~np.isnan(q)]
            )
            quant_fixed.append(q)
    return np.array(quant_fixed).T


def make_bndfile(output_path, t0, tf, et, tide, ntr, Hs):
    """ 
    Create water level boundary file using the tide, ntr, and wave height.
    Standard PS-CoSMoS practice is: eta [surface elevation] = tide + ntr + 0.2 * Hs (to represent setup)

    @params
        output_path - "/path/to/sfincs.bzs"
        t0 - start time [datetime64]
        tf - end time [datetime64]
        et - timestamps for waterlevel parameters
        tide, ntr, Hs -- waterlevel parameter, shapes = (time,stations)
    """
    # Obtain eta
    tok = np.logical_and(et >= t0, et <= tf)
    eta = tide[tok] + ntr[tok] + 0.2 * Hs[tok]

    # Fix eta of any remaining nans
    eta = nan_interp(eta, et[tok])

    # Save
    time_seconds = (et[tok] - t0) / np.timedelta64(1, 's')
    out = np.column_stack([time_seconds, eta])
    fmt = ["%.1f"] + ["%.3f"] * eta.shape[1]
    np.savetxt(output_path, out, fmt=fmt)
    

def make_bndfile_synthetic(output_path, t0, tf, tide_year, et, tide, ntr, Hs):
    """
    as above, but now make a synthetic year by using tide from a different year
    Everything else is the same. 

    @params
        tide_year - int year to use for tide
    """
    # Get tide year as a delta
    dt_years = t0.astype('datetime64[Y]') - np.datetime64(str(tide_year))
    n_years = dt_years.astype(int)
    n_seconds = int(n_years*365.25*24*3600)
    dt = np.timedelta64(n_seconds, 's')

    # As before, obtain eta, but mix and match years
    normal_tok = np.logical_and(et >= t0, et <= tf)
    tide_year_tok = np.logical_and(et >= t0-dt, et <= tf-dt)
    eta = tide[tide_year_tok] + ntr[normal_tok] + 0.2 * Hs[normal_tok]

    # Fix eta of any remaining nans
    eta = nan_interp(eta, et[normal_tok])

    # Save
    time_seconds = (et[normal_tok] - t0) / np.timedelta64(1, 's')
    out = np.column_stack([time_seconds, eta])
    fmt = ["%.1f"] + ["%.3f"] * eta.shape[1]
    np.savetxt(output_path, out, fmt=fmt)

if __name__ == '__main__':

    # Note scenarios we intend to run
    scenarios = ['reanalysis', '000', '025', '050', '100', '150', '200', '300']

    # Note years available in reanalysis dataset
    years = np.arange(1942, 2024, 1)

    # Obtain output directories
    parent_dir = '/home/cassandra/Snohomish/2026-10-08-demo/'
    parent_subdirs = []
    for s in scenarios:
        parent_subdirs.append(parent_dir + s + '/')
        if not os.path.exists(parent_subdirs[-1]):
            os.mkdir(parent_subdirs[-1])

    # Obtain the relevant DFM output files
    dfm_dir = '/media/cassandra/Expansion/dfm_release/'
    wave_files = {}
    waterlevel_files = {}
    for s in ['000', '025', '050', '100', '150', '200', '300']:
        wave_files.update(      {s:dfm_dir + 'Reanalysis_and_Projected_CoSMoSwaves_Snohomish_sealevel'       + s + 'm.nc'})
        waterlevel_files.update({s:dfm_dir + 'Reanalysis_and_Projected_CoSMoSwaterlevels_snohomish_sealevel' + s +  '.nc'})

    # Sample one of them for x,y point to set sfincs.bnd file
    # Doesn't matter which one
    waterlevel_xy = xy_from_waterlevel(waterlevel_files['000'])
    xy_to_file(waterlevel_xy, parent_dir + 'sfincs.bnd')

    # Same for WFLOW but there's just one file
    wflow_dir = '/media/cassandra/Expansion/WFLOW_release/snoho/'
    discharge_file   = wflow_dir + 'Reanalysis_and_Projected_WFLOWdischarges_snohomish.nc'

    # Note also the wflow remapping
    # This is a manually generated shapefile with 2-point linestrings, used for selecting which WFLOW outputs to 
    # use, and exactly where to put them. WFLOW output points are not perfectly aligned to our DEM, so at minimum
    # they must be remapped to the actual channels. 
    remap_file = parent_dir + 'discharge_remap.geojson'

    # Load the discharge
    qt, Q, Qdiff, discharge_xy = load_discharge(discharge_file, remap_file)

    # Make sfincs.src file from the xy points
    xy_to_file(discharge_xy, parent_dir + 'sfincs.src')

    # Need template sfincs.inp file that we can format to whatever we need
    template_file = parent_dir + 'sfincs_template.inp'

    # Buffer for spinup time. 1 week. 
    buffer = np.timedelta64(7*24*60, 'm')

    # Iterate through scenarios and years to create model inputs
    for s in scenarios:
        print('\n'+s,flush=True) # Progress meter

        # Use Python built-in garbage collector to keep memory usage under control
        # This automatically happens in many cases but good to force it periodically in this kind of file.
        gc.collect()

        if s == 'reanalysis':
            # Start by loading waterlevel & wave data into memory
            # In the case of reanalysis, doesn't matter which one we load
            waterlevels = xr.open_dataset(waterlevel_files['000'])
            waves       = xr.open_dataset(wave_files['000'])
            et   = waterlevels.time.values
            wl   = waterlevels.waterlevel.values
            ntr  = waterlevels.ntr.values
            hs   = waves.Hs.values
            tide = wl - ntr

            for yi, y in enumerate(years):
                print('.',end='',flush=True) # Progress meter

                # Define water year time bound
                t0 = np.datetime64(str(y-1) + '-10-01T00:00') - buffer
                tf = np.datetime64(str(y)   + '-10-01T00:00')

                # Make model subdirectory as needed
                model_dir = parent_dir + s + '/' + 'S' + "{:03d}".format(yi) + '/'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                # Make model input files
                make_disfile(model_dir + 'sfincs.dis', t0, tf, qt, Q)
                make_bndfile(model_dir + 'sfincs.bnd', t0, tf, et, tide, ntr, hs)
                make_inpfile(template_file, model_dir + 'sfincs.inp', t0, tf)


        else:
            # Start by loading waterlevel & wave data into memory
            # Make sure to load the right scenario
            waterlevels = xr.open_dataset(waterlevel_files[s])
            waves       = xr.open_dataset(wave_files[s])
            et   = waterlevels.time.values
            wl   = waterlevels.waterlevel.values
            ntr  = waterlevels.ntr.values
            hs   = waves.Hs.values
            tide = wl - ntr

            # Get the mean cmip6 diffs and nan fix
            wldiff = np.nanmean(waterlevels.wl_CmipDiff.values, axis=2) # default shape = (time, station, cmip6)
            wldiff = nan_fix_wrapper(wl, wldiff, et)
            hsdiff = np.nanmean(waves.hs_CmipDiff.values, axis=0) # default shape = (cmip6, time, station)
            hsdiff = nan_fix_wrapper(hs, hsdiff, et)

            for yi, y in enumerate(years):
                print('.',end='',flush=True) # Progress meter

                # Define water year time bound
                t0 = np.datetime64(str(y-1) + '-10-01T00:00') - buffer
                tf = np.datetime64(str(y)   + '-10-01T00:00')

                # Make model subdirectory as needed
                model_dir = parent_dir + s + '/' + 'S' + "{:03d}".format(yi) + '/'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                # Make model input files
                # Make sure to apply cmip6 diff
                make_disfile(model_dir + 'sfincs.dis', t0, tf, qt, Q + Qdiff)
                make_bndfile(model_dir + 'sfincs.bnd', t0, tf, et, tide, ntr + wldiff, hs + hsdiff)
                make_inpfile(template_file, model_dir + 'sfincs.inp', t0, tf)
                    
            # Synthetic years. Need 20 on top of the 80 we just made
            draw_years = np.arange(1942, 2020, 1)
            drawn = random.sample(sorted(draw_years), 40)
            NTR_years = drawn[:20]
            TIDE_years = drawn[20:]
            for i, yi in enumerate(np.arange(len(years), len(years) + 20, 1)):
                print(',',end='',flush=True) # Progress meter

                tide_year = TIDE_years[i]
                t0 = np.datetime64(str(NTR_years[i]-1) + '-10-01T00:00') - buffer
                tf = np.datetime64(str(NTR_years[i])   + '-10-01T00:00')

                # Make model subdirectory as needed
                model_dir = parent_dir + s + '/' + 'S' + "{:03d}".format(yi) + '/'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                # Make model input files
                # Make sure to apply cmip6 diff
                # Make sure to use the tide year for synthetic bnd file
                make_disfile(model_dir + 'sfincs.dis', t0, tf, qt, Q + Qdiff)
                make_bndfile_synthetic(model_dir + 'sfincs.bnd', t0, tf, tide_year, et, tide, ntr + wldiff, hs + hsdiff)
                make_inpfile(template_file, model_dir + 'sfincs.inp', t0, tf)

                # Note model params
                with open(model_dir + 'year_info.txt', 'w') as f:
                    f.write(str(model_dir) + 
                            ', NTR YEAR: ' + str(NTR_years[i]) + 
                            ', TIDE YEAR: ' + str(TIDE_years[i]))