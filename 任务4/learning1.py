import matplotlib.pyplot as plt
import numpy as np


def visualize_longest_white_column():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Settings
    road_color = 'black'  # Background
    track_color = 'white'  # The road
    lwc_color = 'red'  # Longest White Column
    center_color = 'lime'  # Image Center

    # Canvas setup
    for ax in [ax1, ax2]:
        ax.set_facecolor(road_color)
        ax.set_xlim(0, 188)  # Standard smart car camera width often around 180-190
        ax.set_ylim(0, 120)
        ax.axis('off')

    # ==========================================
    # Scenario 1: Crossroad (Symmetric)
    # ==========================================
    ax1.set_title("Scenario 1: Crossroad\n(Symmetric)", color='black', fontsize=14)

    # Draw Vertical Road (Main straight)
    ax1.fill_between([70, 118], 0, 120, color=track_color)
    # Draw Horizontal Road (Crossing)
    ax1.fill_between([0, 188], 50, 80, color=track_color)

    # Calculate Center
    img_center = 188 / 2

    # Longest White Column Visualization
    # In a crossroad, the longest view is straight ahead, right in the middle
    lwc_x = 94

    # Draw Image Center Line
    ax1.axvline(x=img_center, color=center_color, linestyle='--', label='Image Center (Index 94)')

    # Draw Longest White Column Arrow
    ax1.arrow(lwc_x, 10, 0, 90, head_width=5, head_length=5, fc=lwc_color, ec=lwc_color, linewidth=3,
              label='Longest White Column')

    ax1.text(94, 5, "LWC Index ≈ 94", color=lwc_color, ha='center', fontweight='bold', fontsize=12,
             backgroundcolor='white')
    ax1.text(94, 110, "Symmetry:\nLWC overlaps Center", color='lime', ha='center', fontsize=10, fontweight='bold')

    # ==========================================
    # Scenario 2: Roundabout / Asymmetric
    # ==========================================
    ax2.set_title("Scenario 2: Roundabout Entry\n(Asymmetric)", color='black', fontsize=14)

    # Draw Road: Straight part on the LEFT, widening to the RIGHT for loop
    # Base straight road
    ax2.fill_between([40, 88], 0, 120, color=track_color)

    # The Bulge (Roundabout entry) on the right side
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    # Creating a bulge shape
    bulge_x = 88 + 40 * np.cos(theta)
    bulge_y = 60 + 40 * np.sin(theta)
    ax2.fill_betweenx(bulge_y, 88, bulge_x, color=track_color)

    # Visual Logic:
    # 1. The Image Center is still 94.
    # 2. The "Geometric Center" of the white blob shifts Right because of the bulge.
    # 3. BUT, the "Longest White Column" (furthest view) usually follows the straight line or the deep entry.
    #    Here we depict it following the straight lane on the left.

    lwc_x_round = 64  # Shifted to the left (the straight lane)

    # Draw Image Center Line
    ax2.axvline(x=img_center, color=center_color, linestyle='--', label='Image Center')

    # Draw Longest White Column Arrow
    ax2.arrow(lwc_x_round, 10, 0, 90, head_width=5, head_length=5, fc=lwc_color, ec=lwc_color, linewidth=3)

    # Annotations
    ax2.text(lwc_x_round, 5, f"LWC Index ≈ {lwc_x_round}", color=lwc_color, ha='center', fontweight='bold', fontsize=12,
             backgroundcolor='white')
    ax2.text(140, 60, "Roundabout Area\n(Widening Right)", color='black', ha='center', fontsize=9)

    # Show the offset
    ax2.annotate('', xy=(img_center, 105), xytext=(lwc_x_round, 105),
                 arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
    ax2.text((img_center + lwc_x_round) / 2, 110, "Significant Offset", color='yellow', ha='center', fontsize=10,
             fontweight='bold')

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_longest_white_column()