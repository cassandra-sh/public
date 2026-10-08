from hydromt_sfincs import SfincsModel
import geopandas as gpd
import pandas as pd


def build_qtr(
        root_dir, DEM_path, MAN_path, 
        include_gdf, refinement_gdf, offshore_gdf, levees_gdf, drainage_gdf, obs_gdf,
        manning_override=0.02, base_res = 80, nr_subgrid_pixels = 20, nr_levels = 20
    ):
    
    sf = SfincsModel(root=root_dir, mode="w+")

    sf.quadtree_grid.create_from_region(
        region={"geom": include_gdf},
        res=base_res,
        crs=32610,
        rotated=False,
        refinement_polygons=refinement_gdf,
    )

    elevation_list = [{"elevation": DEM_path}]
    sf.quadtree_elevation.create(
        elevation_list=elevation_list,
        interp_method="linear",
        buffer_cells=1,
    )

    sf.quadtree_mask.create_active(
        include_polygon=include_gdf,
    )

    sf.quadtree_mask.create_boundary(
        btype="waterlevel",
        include_polygon=offshore_gdf,
        reset_bounds=True,
    )

    roughness_list = [{"manning": MAN_path}]
    sf.quadtree_roughness.create(
        roughness_list=roughness_list,
        manning_land=manning_override,
        manning_sea=manning_override,
        rgh_lev_land=0,   # z = 0 m as land/sea boundary
    )

    sf.weirs.create(
        locations=levees_gdf,
        dep=DEM_path,   # sample crest elevation from DEM
        merge=False,
    )

    sf.drainage_structures.create(
        locations=drainage_gdf,
        merge=False,
    )

    sf.observation_points.create(
        locations=obs_gdf,
        merge=False,
    )

    sf.quadtree_subgrid.create(
        elevation_list=elevation_list,
        roughness_list=roughness_list,
        manning_land=manning_override,
        manning_water=manning_override,
        manning_level=0.0, # z = 0 m as land/sea boundary
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels, 
        write_dep_tif=False,
        write_man_tif=False,
        nrmax=2000,
    )

    sf.write()

    # Sanity Checks
    fig, ax = sf.plot_basemap(
        variable="elevation", plot_region=False, plot_geoms = False, 
        plot_bounds=True, bmap="sat", vmin=-5, vmax=15,
    )
    fig.savefig(root_dir + 'elevation.png')

    fig, ax = sf.plot_basemap(
        variable="manning", plot_region=False, plot_geoms = False, plot_bounds=True, 
        bmap="sat", vmin=0.02, vmax=0.04,
    )  
    fig.savefig(root_dir + 'manning.png')
    

if __name__ == '__main__':
    # Can download sample shape files at 
    # https://drive.google.com/drive/folders/1ffh0b2s4rtDE-1UQ8RzZgIIkygZfWpp5?usp=sharing
    parent_dir = '/home/cassandra/Snohomish/2026-10-08-demo/'

    DEM_path = "/home/cassandra/Data/Snoh_DEM_composite/Snohomish_MosaicDEM_modded.tif"
    MAN_path = parent_dir + 'snohomish_manning.tif'

    # Polygon which defines the model domain
    include_gdf  = gpd.read_file(parent_dir + "include.geojson")

    # Polygon which defines which edges of the domain are for waterlevel boundaries
    # This is therefore where the offshore boundary condition lies
    offshore_gdf = gpd.read_file(parent_dir + "offshore_boundary.geojson")

    # Linestrings for levees
    levees_gdf   = gpd.read_file(parent_dir + "levees.geojson")

    # Drainage infrastructure. Works with columns par1 (float), name (string), type (int = 1,2 or 3)
    # and 2 point LineString geometry (input point, output point)
    drainage_gdf = gpd.read_file(parent_dir + "drainage.geojson")

    # Formatted with columns x,y,name,and geometry(=Point(x,y))
    obs_gdf      = gpd.read_file(parent_dir + "obs.geojson")

    # Formattiing refinement polygons as HydroMT likes it
    # Base resolution is 80 by default so refinement levels are
    # 1: 40 m
    # 2: 20 m
    # 3: 10 m
    refine_specs = [
        (parent_dir + "refine_20m.geojson", 2),
        (parent_dir + "refine_10m.geojson", 3)]
    refine_parts = []
    for fpath, level in refine_specs:
        gdf = gpd.read_file(fpath)
        gdf = gdf[["geometry"]].copy()
        gdf["refinement_level"] = level
        refine_parts.append(gdf)
    refinement_gdf = gpd.GeoDataFrame(
        pd.concat(refine_parts, ignore_index=True), crs=include_gdf.crs
    )

    # Build
    build_qtr(
        parent_dir, DEM_path, MAN_path, 
        include_gdf, refinement_gdf, offshore_gdf, levees_gdf, drainage_gdf, obs_gdf
    )