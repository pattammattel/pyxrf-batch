import tifffile as tf
import glob,os,tqdm, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path




def pyxrf_output_tiffs_to_image(wd, search_string ='detsum*quant*.tiff',
                             plot_norm_cbar = False,remove_edges = False, 
                             colormap= 'viridis',
                             scalebar= False,   
                             scalebar_params ={'color':'w','pixel_size':0.120,
                            'unit': 'um', 'font_size':15,
                            'length':6, "thickness":0.02,
                            'location':'lower left'},
                            label_xrf = True, norm_xrf_with_sum = False,
                            elem_line_str = (1,2)):
    
    xrf_tiff_files = [path for path in Path(wd).rglob(f'{search_string}')]
    #print(xrf_tiff_files[0])

    save_to_png = os.path.join(wd, "png_images")
    save_to_svg = os.path.join(wd, "svg_images")

    if not os.path.exists(save_to_png):
        os.makedirs(save_to_png)

    if not os.path.exists(save_to_svg):
        os.makedirs(save_to_svg)

    #color_schemes = [colormap for i in range(len(xrf_tiff_files))]

    for i, im_path in enumerate(tqdm.tqdm(xrf_tiff_files)):
        
        image = tf.imread(im_path)
        vsize, hsize = np.shape(image)

        if norm_xrf_with_sum:
            sum_tiff = tf.imread(os.path.join(image_path.parent, "SUM_maps_XRF_total_cnt.tiff"))

        if remove_edges:
            image = image[2:-2,2:-2]
            sum_tiff = sum_tiff[2:-2,2:-2]

        # Plot the image
        fig, ax = plt.subplots(figsize =(7,7))


        ax.imshow(image, cmap=eval(f"cm.{colormap}"))
        # Create a ScalarMappable object for the colorbar
        mappable = cm.ScalarMappable(cmap=eval(f"cm.{colormap}"))
        cmap_name = colormap
        ax.axis('off')  # Remove axis marks

        if plot_norm_cbar:
            mappable.set_array(image/np.nanmax(image))
        elif norm_xrf_with_sum:
            mappable.set_array(image/sum_tiff)
        else:
            mappable.set_array(image)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add color bar
        cbar = plt.colorbar(mappable, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=8)  # Adjust color bar tick label size

        str_ = im_path.stem.split('_')

        try:
            elem_name = f"{str_[elem_line_str[0]]}-{str_[elem_line_str[1]]}"

        except IndexError:
            elem_name = f"{str_[-1]}"


        if scalebar:
            logfile = os.path.join(im_path.parent, "maps_log_tiff.txt")
            resolution = scalebar_params['pixel_size']
            scalebar_pixels = np.ceil(scalebar_params['length']/resolution)
            #scalebar_pixels = np.ceil(hsize*0.2)
            label = f"{int(np.around(scalebar_pixels*resolution, -2))} {scalebar_params['unit']}"
            
            scalebar = AnchoredSizeBar(ax.transData, 
                                        scalebar_pixels,
                                        label, 
                                        loc=scalebar_params['location'], 
                                        pad=0.2,
                                        color = scalebar_params['color'],
                                        label_top = True,
                                        frameon = False,
                                        size_vertical = np.ceil(vsize*scalebar_params['thickness']),
                                        fontproperties = fm.FontProperties(size=30)
                                        )
            ax.add_artist(scalebar)

        if label_xrf:

            ax.text(int(vsize*0.08), int(hsize*0.08), 
                    elem_name, color='w', fontsize=30, fontweight='bold')

        # Save the plot as PNG
        output_path_png = os.path.join(save_to_png,(os.path.splitext(im_path.stem)[0]+f"_{cmap_name}"))
        output_path_svg = os.path.join(save_to_svg,(os.path.splitext(im_path.stem)[0]+f"_{cmap_name}"))
        plt.savefig(output_path_png+".png", dpi=600, bbox_inches='tight')
        plt.savefig(output_path_svg+".svg", dpi=600, bbox_inches='tight')

        # Show the plot (optional)
        plt.close()
        
def batch_img_conversion(wd,search_string ='detsum*norm*.tiff'):
    xrf_dirs = glob.glob(os.path.abspath(wd)+"/output*")

    for fldr in xrf_dirs:
        print(fldr)
        pyxrf_output_tiffs_to_image(fldr,
                                    search_string = search_string)
        

def export_scan_params(sid=-1, zp_flag=True, save_to=None, real_test=0):
    """
    Fetch scan parameters, ROI positions, step size, and the full start_doc
    for scan `sid`.  Optionally write them out as JSON.

    Returns a dict with:
      - scan_id
      - start_doc
      - roi_positions
      - step_size (computed from scan_input for 2D_FLY_PANDA)
    """
    if real_test == 0:
        print("[EXPORT] Skipping scan params export in test mode.")
        return
    # 1) Pull the header
    hdr = db[int(sid)]
    start_doc = dict(hdr.start)  # cast to plain dict

    # 2) Grab the baseline table and build the ROI dict
    tbl = db.get_table(hdr, stream_name='baseline')
    row = tbl.iloc[0]
    if zp_flag:
        roi = {
            "zpssx":    float(row["zpssx"]),
            "zpssy":    float(row["zpssy"]),
            "zpssz":    float(row["zpssz"]),
            "smarx":    float(row["smarx"]),
            "smary":    float(row["smary"]),
            "smarz":    float(row["smarz"]),
            "zp.zpz1":  float(row["zpz1"]),
            "zpsth":    float(row["zpsth"]),
            "zps.zpsx": float(row["zpsx"]),
            "zps.zpsz": float(row["zpsz"]),
        }
    else:
        roi = {
            "dssx":  float(row["dssx"]),
            "dssy":  float(row["dssy"]),
            "dssz":  float(row["dssz"]),
            "dsx":   float(row["dsx"]),
            "dsy":   float(row["dsy"]),
            "dsz":   float(row["dsz"]),
            "sbz":   float(row["sbz"]),
            "dsth":  float(row["dsth"]),
        }

    # 3) Compute unified step_size from scan_input
    scan_info = start_doc.get("scan", {})
    si = scan_info.get("scan_input", [])
    if scan_info.get("type") == "2D_FLY_PANDA" and len(si) >= 3:
        fast_start, fast_end, fast_N = si[0], si[1], si[2]
        step_size = abs(fast_end - fast_start) / fast_N
    else:
        raise ValueError(f"Cannot compute step_size for scan type {scan_info.get('type')}")

    # 4) Assemble the result dict
    result = {
        "scan_id":       int(sid),
        "start_doc":     start_doc,
        "roi_positions": roi,
        "step_size":     float(step_size),
    }

    # 5) Optionally write out JSON
    if save_to:
        if os.path.isdir(save_to):
            filename = os.path.join(save_to, f"scan_{sid}_params.json")
        else:
            filename = save_to if save_to.lower().endswith(".json") else save_to + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

    return result


def export_batch_scan_params(scan_ids, zp_flag=True, save_to=None, real_test=0):
    """
    Export scan parameters for a batch of scan IDs.
    
    Args:
        scan_ids (list): List of scan IDs to export
        zp_flag (bool): Whether to use ZP motors or DS motors
        save_to (str): Directory or base filename to save to
        real_test (int): If 0, skip actual export (test mode)
    
    Returns:
        dict: Dictionary mapping scan_id to exported parameters
    """
    if real_test == 0:
        print(f"[EXPORT] Skipping batch scan params export in test mode for {len(scan_ids)} scans.")
        return {}
    
    results = {}
    
    for i, sid in enumerate(scan_ids):
        print(f"[BATCH] Exporting scan {sid} ({i+1}/{len(scan_ids)})")
        try:
            # Determine save path for this scan
            scan_save_to = None
            if save_to:
                if os.path.isdir(save_to):
                    scan_save_to = save_to
                else:
                    # If save_to is a filename, create directory structure
                    base_dir = os.path.dirname(save_to) or "."
                    scan_save_to = base_dir
            
            result = export_scan_params(
                sid=sid,
                zp_flag=zp_flag,
                save_to=scan_save_to,
                real_test=real_test
            )
            
            if result:
                results[sid] = result
                print(f"[BATCH] ✅ Exported scan {sid}")
            else:
                print(f"[BATCH] ⚠️ No data returned for scan {sid}")
                
        except Exception as e:
            print(f"[BATCH] ❌ Error exporting scan {sid}: {e}")
            results[sid] = {"error": str(e)}
    
    # Optionally save a summary file
    if save_to and results:
        summary_path = os.path.join(save_to if os.path.isdir(save_to) else os.path.dirname(save_to), 
                                   "batch_export_summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump({
                    "exported_scans": list(results.keys()),
                    "total_scans": len(scan_ids),
                    "successful_exports": len([r for r in results.values() if "error" not in r]),
                    "failed_exports": len([r for r in results.values() if "error" in r]),
                    "export_timestamp": time.time()
                }, f, indent=2)
            print(f"[BATCH] Summary saved to: {summary_path}")
        except Exception as e:
            print(f"[BATCH] ⚠️ Could not save summary: {e}")
    
    print(f"[BATCH] Completed batch export: {len(results)} scans processed")
    return results




def save_all_scan_params(parent_dir, zp_flag=True, real_test=1, skip_existing=True):
    """
    Batch export scan parameter JSON files for all scan result folders
    under a given parent directory.

    Example directory structure:
        /Ajith_2025Q1/NbPt/xrfs/
            ├─ output_tiff_scan2D_324559/
            ├─ output_tiff_scan2D_324560/
            ├─ output_tiff_scan2D_324561/

    Each output_tiff_scan2D_<sid> folder will get:
        scan_<sid>_params.json

    Parameters
    ----------
    parent_dir : str
        Parent directory containing multiple scan result folders.
    zp_flag : bool, optional
        Whether to extract ZP or DS ROI positions.
    real_test : int, optional
        If 0, run in dry-run mode (does not save files).
    skip_existing : bool, optional
        If True, skip scans that already have a JSON file.

    Returns
    -------
    list
        List of JSON file paths created.
    """
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    subdirs = sorted(
        [d for d in os.listdir(parent_dir) if d.startswith("output_tiff_scan2D_")]
    )
    if not subdirs:
        print(f"[INFO] No output_tiff_scan2D_* folders found under {parent_dir}")
        return []

    exported = []
    print(f"[EXPORT] Found {len(subdirs)} scan folders under {parent_dir}")

    for name in subdirs:
        scan_path = os.path.join(parent_dir, name)
        try:
            sid = int(name.split("_")[-1])
        except ValueError:
            print(f"⚠️  Skipping invalid folder name: {name}")
            continue

        json_path = os.path.join(scan_path, f"scan_params.json")

        if skip_existing and os.path.exists(json_path):
            print(f"[SKIP] JSON already exists for scan {sid}")
            continue

        try:
            export_scan_params(sid=sid, zp_flag=zp_flag, save_to=json_path, real_test=real_test)
            exported.append(json_path)
            print(f"[OK] Exported params for scan {sid}")
        except Exception as e:
            print(f"[FAIL] Could not export scan {sid}: {e}")

    print(f"[DONE] Exported {len(exported)} JSON files.")
    return exported





import os
import json
import pandas as pd
import numpy as np

def merge_scan_json_to_csv(parent_dir, overwrite=False, round_decimals=4):
    """
    Collect all existing scan_XXXX_params.json files from each
    output_tiff_scan2D_* directory under parent_dir and compile them
    into a single CSV summary file saved to parent_dir.

    Parameters
    ----------
    parent_dir : str
        Path containing scan result directories.
    overwrite : bool, optional
        If True, overwrite existing CSV file instead of appending.
    round_decimals : int or None, optional
        Number of decimal places to round numeric values to.
        Set to None to disable rounding.

    Returns
    -------
    str
        Path to the CSV file created or updated.
    """
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    subdirs = sorted(
        [d for d in os.listdir(parent_dir) if d.startswith("output_tiff_scan2D_")]
    )
    if not subdirs:
        print(f"[INFO] No scan folders found in {parent_dir}")
        return None

    rows = []
    print(f"[MERGE] Scanning {len(subdirs)} subfolders for JSON files...")

    for name in subdirs:
        scan_dir = os.path.join(parent_dir, name)
        for fname in os.listdir(scan_dir):
            if fname.startswith("scan_") and fname.endswith("_params.json"):
                json_path = os.path.join(scan_dir, fname)

                # Skip empty JSON files
                if os.path.getsize(json_path) == 0:
                    print(f"[SKIP] Empty JSON file: {json_path}")
                    continue

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    sid = data.get("scan_id")
                    step = data.get("step_size", "")
                    roi = data.get("roi_positions", {})
                    start_doc = data.get("start_doc", {})
                    plan = start_doc.get("plan_name", "")
                    sample = start_doc.get("sample_name", "")
                    scan_type = start_doc.get("scan", {}).get("type", "")

                    row = {
                        "scan_id": sid,
                        "plan_name": plan,
                        "sample_name": sample,
                        "scan_type": scan_type,
                        "step_size": step,
                    }
                    row.update(roi)
                    rows.append(row)

                except Exception as e:
                    print(f"[FAIL] Could not read {json_path}: {e}")

    if not rows:
        print("[INFO] No valid JSON files found.")
        return None

    # Create dataframe
    df = pd.DataFrame(rows).drop_duplicates(subset=["scan_id"])

    # Round numeric columns if requested
    if round_decimals is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(round_decimals)

    csv_path = os.path.join(parent_dir, "scan_params_summary.csv")

    # Save or update
    if not overwrite and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["scan_id"])
        combined.to_csv(csv_path, index=False)
        print(f"[UPDATE] Appended new data to {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] Wrote {len(df)} entries to {csv_path}")

    return csv_path





def save_all_scan_params_to_csv(parent_dir, zp_flag=True, real_test=1, overwrite=False):
    """
    Collect scan parameters for all output_tiff_scan2D_* folders under `parent_dir`
    and save a summary CSV file in that same directory.

    Each row corresponds to a scan and includes:
      - scan_id
      - step_size
      - ROI motor positions
      - optional metadata from the start_doc (plan_name, sample_name, etc.)

    Parameters
    ----------
    parent_dir : str
        Path containing the output_tiff_scan2D_* folders.
    zp_flag : bool, optional
        Whether to use ZP or DS motor positions.
    real_test : int, optional
        If 0, dry-run mode (no saving).
    overwrite : bool, optional
        If False, will append to an existing CSV if found.

    Returns
    -------
    str
        Full path to the saved CSV file.
    """
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")

    subdirs = sorted(
        [d for d in os.listdir(parent_dir) if d.startswith("output_tiff_scan2D_")]
    )
    if not subdirs:
        print(f"[INFO] No output_tiff_scan2D_* folders found under {parent_dir}")
        return None

    rows = []
    print(f"[EXPORT] Collecting scan parameters from {len(subdirs)} folders...")

    for name in subdirs:
        try:
            sid = int(name.split("_")[-1])
        except ValueError:
            print(f"⚠️  Skipping invalid folder: {name}")
            continue

        try:
            result = export_scan_params(sid=sid, zp_flag=zp_flag, real_test=real_test)
            roi = result["roi_positions"]
            start_doc = result["start_doc"]

            # Flatten main fields into one dict
            row = {
                "scan_id": result["scan_id"],
                "step_size": result["step_size"],
                "plan_name": start_doc.get("plan_name", ""),
                "sample_name": start_doc.get("sample_name", ""),
                "scan_type": start_doc.get("scan", {}).get("type", ""),
            }
            row.update(roi)
            rows.append(row)

        except Exception as e:
            print(f"[FAIL] Could not fetch scan {sid}: {e}")

    if not rows:
        print("[INFO] No valid scan data found.")
        return None

    df = pd.DataFrame(rows)
    csv_path = os.path.join(parent_dir, "scan_params_summary.csv")

    if not overwrite and os.path.exists(csv_path):
        # Append new rows (avoids duplicates)
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["scan_id"])
        combined.to_csv(csv_path, index=False)
        print(f"[UPDATE] Appended new data to {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] Wrote {len(df)} entries to {csv_path}")

    return csv_path
