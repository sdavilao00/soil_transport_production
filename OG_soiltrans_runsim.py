# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:47:29 2025

@author: 12092
"""

def run_simulation(in_tiff, hollow_shapefile, K, Sc, dt, target_time):
    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z_old = grid.at_node['topographic__elevation']
    
    # Load the hollow points shapefile
    hollow_points = gpd.read_file(hollow_shapefile)
    
    # Create an 8m buffer around each point
    hollow_buffer = hollow_points.buffer(8)
    
    # Convert buffer to a mask
    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()
        transform = src.transform
        grid_shape = (meta['height'], meta['width'])
        buffer_mask = geometry_mask([geom for geom in hollow_buffer], transform=transform, invert=True, out_shape=grid_shape)
    
    # Convert 2D mask to 1D landlab node structure
    buffer_mask_flat = buffer_mask.flatten()
    
    # Ensure mask matches landlab's node count
    if buffer_mask_flat.shape[0] > grid.number_of_nodes:
        buffer_mask_flat = buffer_mask_flat[:grid.number_of_nodes]
    
    # Initialize soil depth: 0m in buffer, 0.5m elsewhere
    soil_depth = np.full(grid.number_of_nodes, 0.5)
    soil_depth[buffer_mask_flat] = 0.0  # Apply the mask
    grid.at_node['soil__depth'] = soil_depth
    
    # Store initial soil depth
    initial_soil_depth = soil_depth.copy()
    total_soil_depth = soil_depth.copy()

    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit)
    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt

        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth/h0)) * dt
        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old

        erosion_exceeds = np.abs(elevation_change) > initial_soil_depth
        if np.any(erosion_exceeds):
            z_new[erosion_exceeds] = z_old[erosion_exceeds] - total_soil_depth[erosion_exceeds]
            total_soil_depth[erosion_exceeds] = production_rate[erosion_exceeds]
            print(f"Erosion exceeded soil depth at time {time} years. Adjusting elevation.")

        change_in_soil_depth = production_rate.copy()
        change_in_soil_depth[~erosion_exceeds] = elevation_change[~erosion_exceeds] + production_rate[~erosion_exceeds]
        total_soil_depth = np.where(erosion_exceeds, production_rate, total_soil_depth + change_in_soil_depth)
        grid.at_node['soil__depth'] = total_soil_depth
        z_old = z_new.copy()

        if time % 50 == 0:
            print(f"Results at {time} years:")
            save_as_tiff(total_soil_depth, os.path.join(OUT_DIRtiff, f"{basefilename}_total_soil_depth_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(production_rate, os.path.join(OUT_DIRtiff, f"{basefilename}_production_rate_{time}yrs.tif"), meta, grid.shape)
            plot_change(total_soil_depth, "Total Soil Depth", basefilename, time, K, grid_shape)
            plot_change(production_rate, "Soil Produced", basefilename, time, K, grid_shape)

        if time % 1000 == 0:
            asc_path = plot_save(grid, z_new, basefilename, time, K, mean_res, XYZunit)
            asc_to_tiff(asc_path, os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_(K={K}).tif"), meta)
    
    os.remove(in_asc)
    os.remove(in_asc.replace('.asc', '.prj'))
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")

