()
        ax.set_title("ArUco Marker Positions (Camera Frame)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        
        for marker_id, matrix in transformation_matrices.items():
            pos = matrix[:3, 3]
            ax.scatter(pos[0], pos[1], pos[2], label=f"ID {marker_id}")
            ax.text(pos[0], pos[1], pos[2], f"{marker_id}", fontsize=8)
        
        ax.legend()
        plt.draw()
        plt.pa