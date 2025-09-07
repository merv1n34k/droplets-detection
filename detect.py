import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import signal

# Enable clean exit on Ctrl+C
signal.signal(signal.SIGINT, signal.SIG_DFL)


def crop_image(img, size):
    """Center crop image to square of given size"""
    h, w = img.shape[:2]
    if h < size or w < size:
        print(f"Warning: Image ({w}x{h}) smaller than crop size ({size}x{size})")
        return img

    # Calculate center crop coordinates
    x = (w - size) // 2
    y = (h - size) // 2
    return img[y : y + size, x : x + size]


def filter_close_circles(circles, min_dist):
    """Remove circles that are too close to each other"""
    if len(circles) <= 1:
        return circles

    # Sort circles by radius (descending) to keep larger circles
    circles = sorted(circles, key=lambda c: c[2], reverse=True)

    keep = []
    for i, circle in enumerate(circles):
        x1, y1, r1 = circle
        should_keep = True

        # Check distance to already kept circles
        for kept_circle in keep:
            x2, y2, r2 = kept_circle
            center_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if center_dist < min_dist:
                should_keep = False
                break

        if should_keep:
            keep.append(circle)

    return keep


def detect_inclusions_improved(
    gray, droplet_circles, min_area=3, max_area=50, threshold_offset=5
):
    """Detect dark inclusions within droplets using adaptive threshold"""
    inclusions = []
    droplet_inclusions = {i: [] for i in range(len(droplet_circles))}

    # Process each droplet individually
    for i, (dx, dy, dr) in enumerate(droplet_circles):
        # Create mask for single droplet with erosion to avoid edges
        single_mask = np.zeros_like(gray)
        # Use 85% of radius to avoid edge artifacts
        eroded_radius = int(dr * 0.85)
        cv2.circle(single_mask, (dx, dy), eroded_radius, 255, -1)

        # Define ROI around droplet
        x1 = max(0, dx - dr - 5)
        y1 = max(0, dy - dr - 5)
        x2 = min(gray.shape[1], dx + dr + 5)
        y2 = min(gray.shape[0], dy + dr + 5)

        # Extract droplet region
        droplet_roi = gray[y1:y2, x1:x2]
        mask_roi = single_mask[y1:y2, x1:x2]

        if droplet_roi.size == 0:
            continue

        # Apply adaptive threshold to find dark spots
        # Lower threshold_offset makes detection more sensitive to dimmer inclusions
        # threshold_offset: 1-3 = very sensitive (detects faint inclusions)
        #                   4-6 = moderate sensitivity (default range)
        #                   7-10 = less sensitive (only sharp inclusions)
        adaptive_thresh = cv2.adaptiveThreshold(
            droplet_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            threshold_offset,  # Block size and constant
        )

        # Invert to get dark regions as white
        dark_spots = cv2.bitwise_not(adaptive_thresh)

        # Apply mask to only keep spots inside the eroded droplet
        dark_spots_masked = cv2.bitwise_and(dark_spots, dark_spots, mask=mask_roi)

        # Find contours of potential inclusions
        contours, _ = cv2.findContours(
            dark_spots_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if min_area < area < max_area:
                # Calculate circularity to filter out elongated shapes
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)

                    # Only keep relatively circular inclusions
                    if circularity > 0.5:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            # Calculate centroid in original image coordinates
                            cx = int(M["m10"] / M["m00"]) + x1
                            cy = int(M["m01"] / M["m00"]) + y1

                            # Double-check inclusion is well inside droplet (not near edge)
                            dist_to_center = np.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
                            if (
                                dist_to_center < dr * 0.8
                            ):  # Must be within 80% of radius
                                inclusions.append((cx, cy))
                                droplet_inclusions[i].append((cx, cy))

    # Create visualization showing detected inclusions
    inclusion_viz = np.zeros_like(gray)
    for cx, cy in inclusions:
        cv2.circle(inclusion_viz, (cx, cy), 3, 255, -1)

    return inclusions, droplet_inclusions, inclusion_viz


def export_to_csv(filename, droplet_circles, droplet_inclusions, px_to_um):
    """Export droplet and inclusion data to CSV"""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Droplet_ID",
                "Center_X_px",
                "Center_Y_px",
                "Radius_px",
                "Diameter_um",
                "Area_um2",
                "Inclusion_Count",
            ]
        )

        for i, (x, y, r) in enumerate(droplet_circles):
            diameter_um = 2 * r * px_to_um
            area_um2 = np.pi * r**2 * (px_to_um**2)
            inclusion_count = len(droplet_inclusions[i])
            writer.writerow([i + 1, x, y, r, diameter_um, area_um2, inclusion_count])


def detect_droplets(args):
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image '{args.image}'")
        return

    # Crop if specified
    if args.crop:
        img = crop_image(img, args.crop)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create figure for visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # 1. Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Image")
    axes[0].axis("off")

    # 2. Median blur
    blurred = cv2.medianBlur(gray, args.blur_size)
    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("2. Median Blur")
    axes[1].axis("off")

    # 3. Canny edges
    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)
    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("3. Canny Edges")
    axes[2].axis("off")

    # 4. Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=args.dp,
        minDist=args.min_dist,
        param1=args.canny_high,
        param2=args.param2,
        minRadius=args.min_diameter // 2,
        maxRadius=args.max_diameter // 2,
    )

    # Filter circles
    valid_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            x, y, r = circle

            # Create temporary mask for circularity check
            temp_mask = np.zeros_like(gray)
            cv2.circle(temp_mask, (x, y), r, 255, -1)

            # Calculate circularity
            contours, _ = cv2.findContours(
                temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                area = cv2.contourArea(contours[0])
                perimeter = cv2.arcLength(contours[0], True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

                # Apply circularity threshold
                if circularity > args.min_circularity:
                    valid_circles.append(circle)

    # Filter circles that are too close
    valid_circles = filter_close_circles(valid_circles, args.min_dist)

    # Detect inclusions using improved method
    inclusions, droplet_inclusions, inclusion_viz = detect_inclusions_improved(
        gray,
        valid_circles,
        args.inclusion_min_area,
        args.inclusion_max_area,
        args.inclusion_threshold,
    )

    # 4. Show detected inclusions
    axes[3].imshow(inclusion_viz, cmap="gray")
    axes[3].set_title("4. Detected Inclusions")
    axes[3].axis("off")

    # Create mask with filtered circles
    mask = np.zeros_like(gray)
    for circle in valid_circles:
        x, y, r = circle
        cv2.circle(mask, (x, y), r, 255, -1)

    # 5. Show mask from detected circles
    axes[4].imshow(mask, cmap="gray")
    axes[4].set_title("5. Droplet Mask")
    axes[4].axis("off")

    # 6. Overlay circles and inclusions on original
    overlay = img.copy()
    if valid_circles:
        for i, (x, y, r) in enumerate(valid_circles):
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            # Draw droplet ID
            cv2.putText(
                overlay,
                str(i + 1),
                (x - 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # Draw inclusions
    for cx, cy in inclusions:
        cv2.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)

    axes[5].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[5].set_title(
        f"6. Droplets (n={len(valid_circles)}) + Inclusions (n={len(inclusions)})"
    )
    axes[5].axis("off")

    # 7. Size distribution
    if valid_circles:
        diameters_px = [2 * r for x, y, r in valid_circles]
        diameters_um = [d * args.px_to_um for d in diameters_px]

        axes[6].hist(
            diameters_um, bins=20, color="skyblue", edgecolor="black", alpha=0.7
        )
        axes[6].set_xlabel("Diameter (μm)")
        axes[6].set_ylabel("Count")
        axes[6].set_title("7. Size Distribution (μm)")
        axes[6].grid(axis="y", alpha=0.3)
    else:
        axes[6].text(
            0.5,
            0.5,
            "No droplets detected",
            ha="center",
            va="center",
            transform=axes[6].transAxes,
        )
        axes[6].axis("off")

    # 8. Inclusion count distribution
    if valid_circles:
        inclusion_counts = [
            len(droplet_inclusions[i]) for i in range(len(valid_circles))
        ]
        total_droplets = len(valid_circles)

        # Create histogram data
        max_inclusions = max(inclusion_counts) if inclusion_counts else 0
        bins = list(range(max_inclusions + 2))  # +2 to include the max value

        counts, bins, patches = axes[7].hist(
            inclusion_counts,
            bins=bins,
            align="left",
            rwidth=0.8,
            color="coral",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(inclusion_counts) / total_droplets * 100,
        )

        axes[7].set_xlabel("Number of Inclusions")
        axes[7].set_ylabel("Percentage of Droplets (%)")
        axes[7].set_title("8. Inclusion Distribution")
        axes[7].grid(axis="y", alpha=0.3)

        # Set integer ticks for x-axis
        axes[7].set_xticks(range(max_inclusions + 1))

        # Add percentage values on top of bars
        for i, patch in enumerate(patches):
            height = patch.get_height()
            if height > 0:
                axes[7].text(
                    patch.get_x() + patch.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    else:
        axes[7].text(
            0.5,
            0.5,
            "No droplets detected",
            ha="center",
            va="center",
            transform=axes[7].transAxes,
        )
        axes[7].axis("off")

    plt.tight_layout()

    # Save plot
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    # Export CSV if requested
    if args.csv and valid_circles:
        export_to_csv(args.csv, valid_circles, droplet_inclusions, args.px_to_um)
        print(f"Data exported to {args.csv}")

    # Minimal output
    print(f"Detected: {len(valid_circles)} droplets, {len(inclusions)} inclusions")


def main():
    parser = argparse.ArgumentParser(
        description="Detect droplets and inclusions in bright field images"
    )

    # Required arguments
    parser.add_argument("image", help="Path to input image")

    # Optional arguments
    parser.add_argument(
        "--blur-size", type=int, default=5, help="Median blur kernel size (default: 5)"
    )
    parser.add_argument(
        "--canny-low", type=int, default=50, help="Canny lower threshold (default: 50)"
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Canny upper threshold (default: 150)",
    )
    parser.add_argument(
        "--min-diameter",
        type=int,
        default=10,
        help="Minimum droplet diameter in pixels (default: 10)",
    )
    parser.add_argument(
        "--max-diameter",
        type=int,
        default=100,
        help="Maximum droplet diameter in pixels (default: 100)",
    )
    parser.add_argument(
        "--min-dist",
        type=int,
        default=20,
        help="Minimum distance between circle centers (default: 20)",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.7,
        help="Minimum circularity threshold (default: 0.7)",
    )
    parser.add_argument(
        "--inclusion-min-area",
        type=int,
        default=3,
        help="Minimum inclusion area in pixels (default: 3)",
    )
    parser.add_argument(
        "--inclusion-max-area",
        type=int,
        default=50,
        help="Maximum inclusion area in pixels (default: 50)",
    )
    parser.add_argument(
        "--inclusion-threshold",
        type=int,
        default=5,
        help="Inclusion detection sensitivity (lower=more sensitive, 1-10, default: 5)",
    )
    parser.add_argument(
        "--px-to-um",
        type=float,
        default=1.0,
        help="Conversion factor: pixels to micrometers (default: 1.0)",
    )
    parser.add_argument(
        "--dp", type=float, default=1, help="HoughCircles dp parameter (default: 1)"
    )
    parser.add_argument(
        "--param2",
        type=int,
        default=30,
        help="HoughCircles accumulator threshold (default: 30)",
    )
    parser.add_argument(
        "--crop",
        type=int,
        help="Center crop to square of given size (e.g., 1024 for 1024x1024)",
    )
    parser.add_argument(
        "--save", type=str, help="Save output to file instead of displaying"
    )
    parser.add_argument("--csv", type=str, help="Export data to CSV file")

    args = parser.parse_args()

    # Validate arguments
    if args.blur_size % 2 == 0:
        args.blur_size += 1

    if args.inclusion_threshold < 1 or args.inclusion_threshold > 10:
        print(f"Warning: inclusion-threshold should be between 1-10, using default 5")
        args.inclusion_threshold = 5

    detect_droplets(args)


if __name__ == "__main__":
    main()
