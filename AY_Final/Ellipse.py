import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import asin, degrees
import re
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit

FILE_PATH = "./AY_Final"                     # Path to file containing images of saturn and header
POINTS_FILE = FILE_PATH + "/pointsFile.npz"  # File containing points for ellipse fits

# Read ORIENTAT from header
def read_orientat(header_path):
    with open(header_path, "r") as f:
        text = f.read()
    m = re.search(r"ORIENTAT\s*=\s*([-+]?\d+\.?\d*)", text)
    if not m:
        raise ValueError("ORIENTAT not found in header")
    return float(m.group(1))


# Rotate image so celestial north is up
def rotate_image(img, angle_deg):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def estimate_ellipse(img_north):
    # Have user select ellipse (automatic fitting performed horribly after lots of attempts)
    #print("Click at least 5 points along the edge of Saturn's rings, then close the window.")
    fig, ax = plt.subplots()
    ax.set_title("Click at least 5 points along the edge of Saturn's rings, then close the window.")
    ax.imshow(img_north)
    points = plt.ginput(n=-1, timeout=0)
    points = np.array(points, dtype=np.float32)
    plt.close(fig)

    if len(points) < 5:
        raise RuntimeError("Please use at least 5 points")
    
    return points
    

def calculate_angles(year, interactive, show_fits):
    IMG_FILE = FILE_PATH + f"/saturn{year}.jpg"       # Saturn image
    HEADER_FILE = FILE_PATH + f"/header{year}.asc"    # FITS header text
    
    # Load image
    img = plt.imread(IMG_FILE)

    # Read ORIENTAT
    orientat = read_orientat(HEADER_FILE)
    #print(f"ORIENTAT = {orientat:.2f}°") # Debug statement

    # Rotate image north-up
    img_north = rotate_image(img, orientat)
    
    # Fit ellipse with OpenCV
    if not interactive:
        npzfile = np.load(POINTS_FILE)
        points = npzfile[f"points_{year}"]
    else:
        points = estimate_ellipse(img_north)
        
    pts = points.reshape(-1, 1, 2)
    ellipse = cv2.fitEllipse(pts)
    (xc, yc), (a, b), theta = ellipse
    a /= 2  # OpenCV axes are full lengths, convert to semi-axes
    b /= 2
    minor = min(a, b)
    major = max(a, b)

    # Compute geometry
    B = degrees(asin(minor / major))
    PA = theta

    print(f"Ring opening angle B = {B:.2f}°")
    print(f"Ring position angle PA = {PA:.2f}°")

    # Plot fitted ellipse
    if show_fits:
        fig, ax = plt.subplots()
        ax.imshow(img_north)
        t = np.linspace(0, 2*np.pi, 300)
        theta_rad = np.radians(theta)
        xs = xc + a*np.cos(t)*np.cos(theta_rad) - b*np.sin(t)*np.sin(theta_rad)
        ys = yc + a*np.cos(t)*np.sin(theta_rad) + b*np.sin(t)*np.cos(theta_rad)
        ax.plot(xs, ys, 'r-', linewidth=2)
        ax.set_title(f"Image from {year}, B = {B:.2f}°, PA = {PA:.2f}°")
        ax.set_axis_off()
        plt.show()
    
    return points, B # So points can be saved to file


if __name__ == "__main__":
    first_year = 1994
    last_year = 2017
    num_observations = 21 # To use less than 21 observations, reduce this number
                          # Using less than 21 observations is recommended if you set interactive to True
    save_data = False     # Set to True to save new ellipse fits of Saturn's rings
    interactive = False   # Set to True to manually fit ellipses of Saturn's rings
    show_fits = False     # Set to True to see all observations' ellipse fits
    
    if num_observations > 21:
        num_observations = 21 # We only have 21 observations, and cannot use more
    
    # Create list of years that excludes the three years missing observations
    years = [y for y in range(first_year, last_year + 1) if y not in (1997, 2006, 2010)] 
    # Choose num_observations semi-evenly spaced observations
    indices = np.linspace(0, len(years)-1, num_observations, dtype=int) 
    selected_years = [years[i] for i in indices]
    
    data = {}
    angles = np.array([])
    
    for year in selected_years:
        print(year)
        points, B = calculate_angles(year, interactive, show_fits)
        if year < 1995 or year > 2009:
            B = B * -1    # Important fo fitting model later. 
                          # Apparent inclinations are calculated as an absolute value, so I adjust that here.
        angles = np.append(angles, B)
        if save_data:
            data[f"points_{year}"] = points
    
    # Save data to .npz file
    if save_data:
        np.savez(FILE_PATH + "/pointsFile2.npz", **data) # Replace "/pointsFile2.npz" with whatever you want the file name to be
    

    # Load the CSV to get precise observation dates
    df = pd.read_csv(FILE_PATH + "/Observation_Metadata.csv")

    # Check column names
    print(df.columns)

    # Extract the date column by header
    date_column = "Observation Start Time (YMDhms)"
    dates_str = df[date_column].astype(str)
    dates = pd.to_datetime(dates_str, format="%Y-%m-%dT%H:%M:%S.%f")

    # Convert to numeric days since first observation
    t0 = dates.iloc[0]
    x = np.array([(d - t0).total_seconds() / (3600*24) for d in dates])  # days as float

    print("x (days):", x)
    
    # Fit sine wave: y = A * sin(B*x + C) + D
    def sin_model(x, A, B, C, D):
        return A * np.sin(B*x + C) + D

    # x: numeric values in days
    # y: Angle
    initial_guess = [np.max(angles), 2*np.pi/(14.5 * 365.25), 0, np.mean(angles)]  # amplitude, freq, phase, offset

    params, _ = curve_fit(sin_model, x, angles, p0=initial_guess)
    y_fit = sin_model(x, *params)
    

    plt.scatter(x, angles, label="data")
    plt.plot(x, y_fit, label="sinusoidal fit")
    plt.axhline(np.min(angles), color="gray", linestyle="--", label=f"min value = {np.min(angles):.2f}")
    plt.axhline(np.max(angles), color="black", linestyle="--", label=f"max value = {np.max(angles):.2f}")
    plt.xlabel("Days since first observation")
    plt.ylabel("Apparent inclination")
    plt.legend()
    plt.show()