import graphviz
import os

# 1. TELL SPYDER WHERE GRAPHVIZ IS (Crucial for Anaconda users)
# This points to the bin folder where the 'dot' executable lives
os.environ["PATH"] += os.pathsep + r'C:\Users\candi\anaconda3\Library\bin\graphviz'

def generate_thesis_flowchart():
    # 2. Initialize the Diagram
    # 'TB' means Top to Bottom flow
    dot = graphviz.Digraph('System_Workflow', comment='Physiological Heat Stress AI')
    dot.attr(rankdir='TB', size='10', nodesep='0.5', ranksep='0.5')
    
    # 3. Define Global Styles
    dot.attr('node', fontname='Arial', fontsize='12')

    # 4. Define Nodes (Shapes based on DFD standards)
    # Start/End: Ovals
    dot.node('START', 'START\n(Initialize Flask & Load .pkl Models)', shape='ellipse', style='filled', fillcolor='#D1EAFF')
    dot.node('END', 'END\n(System Ready for Next Patient)', shape='ellipse', style='filled', fillcolor='#D1EAFF')

    # Input/Output: Parallelograms
    dot.node('INPUT', 'INPUT\n(11 Physiological Parameters)', shape='parallelogram', style='filled', fillcolor='#FFF4D1')
    dot.node('OUTPUT', 'OUTPUT\n(Diagnosis: Normal vs Hyperthermic)', shape='parallelogram', style='filled', fillcolor='#FFF4D1')

    # Processes: Rectangles
    dot.node('VALIDATE', 'Data Validation\n(Type & Range Check)', shape='box', style='rounded')
    dot.node('SCALE', 'Preprocessing\n(StandardScaler Normalization)', shape='box', style='rounded')
    dot.node('INFER', 'AI Inference Engine\n(Gradient Boosting Regressor)', shape='box', style='rounded')

    # Decision: Diamond
    dot.node('DECISION', 'Is Predicted Tcore\n>= 38.0°C?', shape='diamond', style='filled', fillcolor='#FFD1D1')

    # 5. Define Connections (The Logic Flow)
    dot.edge('START', 'INPUT')
    dot.edge('INPUT', 'VALIDATE')
    dot.edge('VALIDATE', 'SCALE')
    dot.edge('SCALE', 'INFER')
    dot.edge('INFER', 'DECISION')
    
    # Decision Labels
    dot.edge('DECISION', 'OUTPUT', label=' YES (Alert)')
    dot.edge('DECISION', 'OUTPUT', label=' NO (Safe)')
    
    dot.edge('OUTPUT', 'END')

    # 6. Save and Render
    # This creates 'Thesis_Flowchart.png' in your Desktop/Heat_Stress_App folder
    output_path = 'Thesis_Flowchart'
    dot.render(output_path, format='png', cleanup=True)
    
    print(f"Success! Your flowchart has been saved as: {output_path}.png")

if __name__ == "__main__":
    try:
        generate_thesis_flowchart()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTip: If you get an 'ExecutableNotFound' error, restart Spyder and try again.")