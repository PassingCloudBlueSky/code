import svgutils.transform as sg
from svgutils.compose import Figure,Panel,SVG,Text,Grid
import sys 

#create new SVG figure
fig = sg.SVGFigure("16cm", "6.5cm")

# load matpotlib-generated figures
fig1 = sg.fromfile('outer_product_visualization.svg')
fig2 = sg.fromfile('output.svg')

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot2.moveto(20, 0, scale_x=0.5)

# add text labels
txt1 = sg.TextElement(25,20, "A", size=12, weight="bold")
txt2 = sg.TextElement(305,20, "B", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2])
fig.append([txt1, txt2])

# save generated SVG files
fig.save("fig_final.svg")

Figure("16cm", "6.5cm", 
        Panel(
              SVG("outer_product_visualization.svg"),
              Text("A", 25, 20, size=12, weight='bold')
             ).move(30,0),
        Panel(
              SVG("output.svg").scale(0.5),
              Text("B", 25, 20, size=12, weight='bold')
             ).move(20, 0),
             Grid(20,20)
        ).save("fig_final_compose.svg")