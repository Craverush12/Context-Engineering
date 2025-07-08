"""
Field Visualization Tools
=========================

Real-time visualization tools for context fields, attractors, resonance patterns,
and field evolution tracking. Supports both static and interactive visualizations.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class FieldVisualizer:
    """
    Comprehensive visualization system for context fields.
    
    Provides multiple visualization modes:
    - Field grid heatmaps
    - Attractor dynamics
    - Resonance patterns  
    - Field evolution over time
    - Interactive exploration
    """
    
    def __init__(self, field_manager=None):
        """
        Initialize field visualizer.
        
        Args:
            field_manager: Optional FieldManager instance for multi-field visualization
        """
        self.field_manager = field_manager
        self.visualization_history = []
        
    def visualize_field_state(self, 
                             context_field,
                             mode: str = 'heatmap',
                             save_path: Optional[str] = None,
                             interactive: bool = False) -> Optional[str]:
        """
        Visualize current field state.
        
        Args:
            context_field: ContextField instance to visualize
            mode: Visualization mode ('heatmap', 'attractors', 'resonance', 'combined')
            save_path: Optional path to save visualization
            interactive: Whether to create interactive visualization
            
        Returns:
            Base64 encoded image string if matplotlib available, else None
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_visualization(context_field, mode)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_visualization(context_field, mode, save_path)
        else:
            return self._create_text_visualization(context_field)
    
    def track_field_evolution(self, 
                             field_states: List[Dict[str, Any]],
                             save_path: Optional[str] = None) -> Optional[str]:
        """
        Create evolution visualization from field state history.
        
        Args:
            field_states: List of field state dictionaries
            save_path: Optional path to save animation
            
        Returns:
            Path to saved animation or base64 string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_evolution_summary(field_states)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Context Field Evolution', fontsize=16)
        
        # Plot evolution metrics
        timestamps = [state.get('timestamp', i) for i, state in enumerate(field_states)]
        
        # Number of elements over time
        element_counts = [state.get('field_properties', {}).get('total_elements', 0) 
                         for state in field_states]
        axes[0, 0].plot(timestamps, element_counts, 'b-o')
        axes[0, 0].set_title('Elements Over Time')
        axes[0, 0].set_ylabel('Number of Elements')
        
        # Number of attractors over time
        attractor_counts = [state.get('field_properties', {}).get('total_attractors', 0)
                           for state in field_states]
        axes[0, 1].plot(timestamps, attractor_counts, 'r-o')
        axes[0, 1].set_title('Attractors Over Time')
        axes[0, 1].set_ylabel('Number of Attractors')
        
        # Field coherence over time
        coherence_scores = [state.get('field_properties', {}).get('coherence', 0)
                           for state in field_states]
        axes[1, 0].plot(timestamps, coherence_scores, 'g-o')
        axes[1, 0].set_title('Field Coherence')
        axes[1, 0].set_ylabel('Coherence Score')
        
        # Field grid evolution (final state)
        if field_states and 'field_grid' in field_states[-1]:
            field_grid = np.array(field_states[-1]['field_grid'])
            im = axes[1, 1].imshow(field_grid, cmap='viridis', origin='lower')
            axes[1, 1].set_title('Final Field State')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return save_path
        else:
            return self._fig_to_base64(fig)
    
    def create_attractor_dynamics_plot(self, 
                                      context_field,
                                      save_path: Optional[str] = None) -> Optional[str]:
        """Create visualization focused on attractor dynamics."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_attractor_summary(context_field)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Attractor positions and strengths
        field_grid = context_field.field_grid
        im1 = ax1.imshow(field_grid, cmap='viridis', alpha=0.6, origin='lower')
        
        attractors = context_field.get_attractors()
        for attractor in attractors:
            x, y = attractor.center
            # Convert to grid coordinates
            grid_x, grid_y = int(x * 99), int(y * 99)
            
            # Draw attractor as circle with size proportional to strength
            circle = patches.Circle((grid_x, grid_y), 
                                  radius=attractor.radius * 100,
                                  fill=False, 
                                  edgecolor='red', 
                                  linewidth=2 + attractor.strength)
            ax1.add_patch(circle)
            
            # Add label
            ax1.text(grid_x, grid_y + attractor.radius * 120, 
                    attractor.name, 
                    ha='center', va='bottom', 
                    color='white', fontweight='bold')
        
        ax1.set_title('Attractor Dynamics')
        ax1.set_xlabel('Field X Coordinate')
        ax1.set_ylabel('Field Y Coordinate')
        plt.colorbar(im1, ax=ax1, label='Field Intensity')
        
        # Plot 2: Attractor strength distribution
        if attractors:
            strengths = [a.strength for a in attractors]
            names = [a.name for a in attractors]
            
            bars = ax2.bar(range(len(strengths)), strengths, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(strengths))))
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45)
            ax2.set_title('Attractor Strengths')
            ax2.set_ylabel('Strength')
            
            # Add value labels on bars
            for bar, strength in zip(bars, strengths):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{strength:.2f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Attractors Detected', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, style='italic')
            ax2.set_title('Attractor Strengths')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return save_path
        else:
            return self._fig_to_base64(fig)
    
    def create_resonance_network_plot(self, 
                                     context_field,
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Create network visualization of resonance patterns."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_resonance_summary(context_field)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        elements = list(context_field.elements.values())
        resonance_patterns = context_field.get_resonance_patterns()
        
        if not elements:
            ax.text(0.5, 0.5, 'No Elements in Field', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, style='italic')
            ax.set_title('Resonance Network')
            return self._fig_to_base64(fig) if not save_path else save_path
        
        # Plot elements as nodes
        for element in elements:
            x, y = element.position
            # Scale position to plot coordinates
            plot_x, plot_y = x * 10, y * 10
            
            # Node size proportional to strength
            node_size = 50 + element.strength * 200
            ax.scatter(plot_x, plot_y, s=node_size, 
                      alpha=0.7, c='blue', edgecolors='black')
            
            # Add element label
            ax.text(plot_x, plot_y + 0.3, element.id[:10], 
                   ha='center', va='bottom', fontsize=8)
        
        # Draw resonance connections
        for pattern in resonance_patterns:
            if len(pattern.participating_elements) >= 2:
                elem_ids = pattern.participating_elements[:2]  # Take first two
                
                # Find element positions
                elem1 = context_field.elements.get(elem_ids[0])
                elem2 = context_field.elements.get(elem_ids[1])
                
                if elem1 and elem2:
                    x1, y1 = elem1.position[0] * 10, elem1.position[1] * 10
                    x2, y2 = elem2.position[0] * 10, elem2.position[1] * 10
                    
                    # Line thickness proportional to resonance strength
                    line_width = 1 + pattern.coherence_score * 3
                    ax.plot([x1, x2], [y1, y2], 
                           color='red', alpha=0.6, linewidth=line_width)
                    
                    # Add resonance strength label
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f'{pattern.coherence_score:.2f}',
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_title(f'Resonance Network ({len(resonance_patterns)} patterns)')
        ax.set_xlabel('Field X Coordinate')
        ax.set_ylabel('Field Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return save_path
        else:
            return self._fig_to_base64(fig)
    
    def _create_interactive_visualization(self, context_field, mode: str) -> str:
        """Create interactive Plotly visualization."""
        if mode == 'heatmap' or mode == 'combined':
            # Create heatmap of field grid
            field_grid = np.array(context_field.field_grid)
            
            fig = go.Figure(data=go.Heatmap(
                z=field_grid,
                colorscale='Viridis',
                showscale=True
            ))
            
            # Add attractors as annotations
            attractors = context_field.get_attractors()
            for attractor in attractors:
                x, y = attractor.center
                grid_x, grid_y = int(x * 99), int(y * 99)
                
                fig.add_annotation(
                    x=grid_x, y=grid_y,
                    text=attractor.name,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="white"
                )
            
            fig.update_layout(
                title="Interactive Context Field Visualization",
                xaxis_title="Field X Coordinate",
                yaxis_title="Field Y Coordinate"
            )
            
            return fig.to_html()
        
        return "<p>Interactive visualization not implemented for this mode</p>"
    
    def _create_static_visualization(self, 
                                   context_field, 
                                   mode: str, 
                                   save_path: Optional[str]) -> Optional[str]:
        """Create static matplotlib visualization."""
        if mode == 'heatmap':
            return self._create_heatmap(context_field, save_path)
        elif mode == 'attractors':
            return self.create_attractor_dynamics_plot(context_field, save_path)
        elif mode == 'resonance':
            return self.create_resonance_network_plot(context_field, save_path)
        elif mode == 'combined':
            return self._create_combined_visualization(context_field, save_path)
        else:
            return self._create_heatmap(context_field, save_path)
    
    def _create_heatmap(self, context_field, save_path: Optional[str]) -> Optional[str]:
        """Create basic field heatmap visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        field_grid = context_field.field_grid
        im = ax.imshow(field_grid, cmap='viridis', origin='lower')
        
        ax.set_title('Context Field Heatmap')
        ax.set_xlabel('Field X Coordinate')
        ax.set_ylabel('Field Y Coordinate')
        plt.colorbar(im, ax=ax, label='Field Intensity')
        
        # Add field statistics as text
        stats_text = f"""Field Statistics:
Elements: {len(context_field.elements)}
Attractors: {len(context_field.attractors)}
Resonance Patterns: {len(context_field.resonance_patterns)}
Coherence: {context_field.measure_field_coherence():.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return save_path
        else:
            return self._fig_to_base64(fig)
    
    def _create_combined_visualization(self, context_field, save_path: Optional[str]) -> Optional[str]:
        """Create comprehensive combined visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main field heatmap (top left, large)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        field_grid = context_field.field_grid
        im = ax_main.imshow(field_grid, cmap='viridis', alpha=0.8, origin='lower')
        
        # Overlay attractors
        attractors = context_field.get_attractors()
        for attractor in attractors:
            x, y = attractor.center
            grid_x, grid_y = int(x * 99), int(y * 99)
            circle = patches.Circle((grid_x, grid_y), 
                                  radius=attractor.radius * 100,
                                  fill=False, edgecolor='red', linewidth=2)
            ax_main.add_patch(circle)
        
        ax_main.set_title('Context Field with Attractors')
        plt.colorbar(im, ax=ax_main, label='Field Intensity')
        
        # Element count over time (top right)
        ax_elements = fig.add_subplot(gs[0, 2])
        ax_elements.bar(['Elements', 'Attractors', 'Resonance'], 
                       [len(context_field.elements), 
                        len(context_field.attractors),
                        len(context_field.resonance_patterns)],
                       color=['blue', 'red', 'green'])
        ax_elements.set_title('Field Components')
        ax_elements.set_ylabel('Count')
        
        # Coherence meter (middle right)
        ax_coherence = fig.add_subplot(gs[1, 2])
        coherence = context_field.measure_field_coherence()
        wedges, texts = ax_coherence.pie([coherence, 1-coherence], 
                                        labels=['Coherent', 'Incoherent'],
                                        colors=['green', 'lightgray'],
                                        startangle=90)
        ax_coherence.set_title(f'Field Coherence\n{coherence:.2f}')
        
        # Resonance network (bottom, full width)
        ax_network = fig.add_subplot(gs[2, :])
        self._plot_resonance_network_subplot(context_field, ax_network)
        
        fig.suptitle('Comprehensive Context Field Analysis', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return save_path
        else:
            return self._fig_to_base64(fig)
    
    def _plot_resonance_network_subplot(self, context_field, ax):
        """Plot resonance network as subplot."""
        elements = list(context_field.elements.values())
        resonance_patterns = context_field.get_resonance_patterns()
        
        if not elements:
            ax.text(0.5, 0.5, 'No Elements in Field', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot elements
        for element in elements:
            x, y = element.position
            ax.scatter(x, y, s=50 + element.strength * 100, 
                      alpha=0.7, c='blue', edgecolors='black')
        
        # Draw resonance connections
        for pattern in resonance_patterns:
            if len(pattern.participating_elements) >= 2:
                elem_ids = pattern.participating_elements[:2]
                elem1 = context_field.elements.get(elem_ids[0])
                elem2 = context_field.elements.get(elem_ids[1])
                
                if elem1 and elem2:
                    x1, y1 = elem1.position
                    x2, y2 = elem2.position
                    ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, 
                           linewidth=1 + pattern.coherence_score * 2)
        
        ax.set_title('Element Resonance Network')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _create_text_visualization(self, context_field) -> str:
        """Create text-based visualization when graphics libraries unavailable."""
        field_state = context_field.get_field_state()
        
        visualization = f"""
Context Field Visualization (Text Mode)
=====================================

Field Properties:
- Total Elements: {len(context_field.elements)}
- Total Attractors: {len(context_field.attractors)}
- Resonance Patterns: {len(context_field.resonance_patterns)}
- Field Coherence: {context_field.measure_field_coherence():.3f}
- Field Age: {time.time() - context_field.creation_time:.1f} seconds

Attractors:
"""
        
        for attractor in context_field.get_attractors():
            visualization += f"- {attractor.name}: strength={attractor.strength:.2f}, elements={len(attractor.elements)}\n"
        
        visualization += "\nResonance Patterns:\n"
        for pattern in context_field.get_resonance_patterns():
            visualization += f"- {pattern.id}: coherence={pattern.coherence_score:.2f}, elements={len(pattern.participating_elements)}\n"
        
        return visualization
    
    def _create_text_evolution_summary(self, field_states: List[Dict[str, Any]]) -> str:
        """Create text summary of field evolution."""
        if not field_states:
            return "No field states provided for evolution analysis."
        
        summary = "Field Evolution Summary\n" + "="*25 + "\n\n"
        
        # Extract metrics over time
        element_counts = [state.get('field_properties', {}).get('total_elements', 0) 
                         for state in field_states]
        attractor_counts = [state.get('field_properties', {}).get('total_attractors', 0)
                           for state in field_states]
        coherence_scores = [state.get('field_properties', {}).get('coherence', 0)
                           for state in field_states]
        
        summary += f"Time Points Analyzed: {len(field_states)}\n"
        summary += f"Element Count: {element_counts[0]} → {element_counts[-1]} (change: {element_counts[-1] - element_counts[0]})\n"
        summary += f"Attractor Count: {attractor_counts[0]} → {attractor_counts[-1]} (change: {attractor_counts[-1] - attractor_counts[0]})\n"
        summary += f"Coherence: {coherence_scores[0]:.3f} → {coherence_scores[-1]:.3f} (change: {coherence_scores[-1] - coherence_scores[0]:.3f})\n"
        
        return summary
    
    def _create_text_attractor_summary(self, context_field) -> str:
        """Create text summary of attractor dynamics."""
        attractors = context_field.get_attractors()
        
        summary = "Attractor Dynamics Summary\n" + "="*26 + "\n\n"
        summary += f"Total Attractors: {len(attractors)}\n\n"
        
        for attractor in attractors:
            summary += f"Attractor: {attractor.name}\n"
            summary += f"  Strength: {attractor.strength:.3f}\n"
            summary += f"  Position: ({attractor.center[0]:.3f}, {attractor.center[1]:.3f})\n"
            summary += f"  Elements: {len(attractor.elements)}\n"
            summary += f"  Age: {time.time() - attractor.formation_time:.1f}s\n\n"
        
        return summary
    
    def _create_text_resonance_summary(self, context_field) -> str:
        """Create text summary of resonance patterns."""
        patterns = context_field.get_resonance_patterns()
        
        summary = "Resonance Patterns Summary\n" + "="*27 + "\n\n"
        summary += f"Total Patterns: {len(patterns)}\n\n"
        
        for pattern in patterns:
            summary += f"Pattern: {pattern.id}\n"
            summary += f"  Coherence: {pattern.coherence_score:.3f}\n"
            summary += f"  Frequency: {pattern.resonance_frequency:.3f}\n"
            summary += f"  Amplitude: {pattern.amplitude:.3f}\n"
            summary += f"  Elements: {len(pattern.participating_elements)}\n\n"
        
        return summary
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def export_field_data(self, context_field, format: str = 'json') -> str:
        """
        Export field data in various formats.
        
        Args:
            context_field: ContextField to export
            format: Export format ('json', 'csv', 'yaml')
            
        Returns:
            Formatted data string
        """
        field_state = context_field.get_field_state()
        
        if format == 'json':
            return json.dumps(field_state, indent=2, default=str)
        elif format == 'csv':
            # Convert to CSV-like format
            csv_lines = ["Field Elements:"]
            csv_lines.append("id,type,content,position_x,position_y,strength")
            
            for element_data in field_state['elements'].values():
                csv_lines.append(f"{element_data['id']},{element_data['type']},{element_data['content'][:50]},"
                               f"{element_data['position'][0]},{element_data['position'][1]},{element_data['strength']}")
            
            csv_lines.append("\nAttractors:")
            csv_lines.append("id,name,center_x,center_y,strength,radius")
            
            for attractor_data in field_state['attractors'].values():
                csv_lines.append(f"{attractor_data['id']},{attractor_data['name']},"
                               f"{attractor_data['center'][0]},{attractor_data['center'][1]},"
                               f"{attractor_data['strength']},{attractor_data['radius']}")
            
            return '\n'.join(csv_lines)
        else:
            return json.dumps(field_state, indent=2, default=str)  # Default to JSON