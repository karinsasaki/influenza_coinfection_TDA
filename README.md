# Topological data analysis of three-stage immune response to influenza-pneumococcal lung infection

In this GitHub page we provide the code that we used in our topological data analysis of co-infection (data) between influenza and bacteria in the lung.


## Files explained

### coinfections.csv file
This is the data set we analyse in this study. This data set was originally published in Duvigneau et al. (2016). In this paper  the authors investigated the hierarchical effects of pro-inflammatory cytokines on the post-influenza susceptibility to pneumococcal co-infection by assessing the early and late kinetics of pro-inflammatory cytokines in the respiratory tract. In the experimental part of this study the mice were divided into three groups and given either a single viral infection (with IAV strain 84 A/PR8/34), a single bacterial infection (S. pneumoniae strain T4) or a co-infection (IAV + T4). The experimental read outs were the bacterial burden, viral titers and cytokine protein concentrations in the lung. They used mathematical modelling that sugested a detrimental role of IFN-gamma alone and in synenergism with IL-6 and TFN-alpha in impaired bacterial clearance. We use the Mapper Algorithm to investigate the global shape of the immune system under the three infection scenarios and we generate a new hypotheses where the immune system goes through three stages and two transition points in its response to co-infection. 

### The functions.py file
This Python script contains all the functions we use for the semi-unsupervised algorithm and for generating the images shown in the paper.


### The generate_images.ipnb file
Here we show how we created the images included in the paper and the supplementary material.

### The generate_images_output folder
Here we include all the images presented in the paper and the supplementary, generated using the generate_images.ipnb.

#### The data_6_cc.csv file
This is a file exported from the generate_images.ipnb jupiter notebook. It is an example of the parameter values for simplicial complexes that have exactly six connected components.

### The parameter_value_grid_search_example.ipynb file
In this Jupiter notebook we show an example of how to run a semi-unsupervised algorithm for finding the parameter values of the Kepler Mapper in order to generate simplicial complexes that are good representations of the data sets and that can be used for further analysis of the biological system.

The specific example we run is all simplicial complexes generated with 

lens1 = distance to the first neighbour

lens2 = distance to the second neighbour

Waring: This script can take a very long time to complete, so we have included the output files in a separate folder (see below).

### The parameter_value_grid_search_example_output folder

#### The all_complexes_parameter_values folder 
Contains all the results of the parameter grid search example above. These are csv files which can be imported into Python as Pandas data frames.

#### The visualise_10_simplest_simplicial_complexes folder 
Contains the html files to visualise the ‘simplex’ simplicial complexes (see the supplementary material for more details).


## Use

Open a Jupyter notebook and import either the parameter_value_grid_search_example.ipynb or the generate_images.ipynb. Run the cells.



# References

Duvigneau S, Sharma-Chawla N, Boianelli A, Stegemann-Koniszewski S, Nguyen VK, Bruder D, Hernandez-Vargas EA. Hierarchical effects of pro-inflammatory cytokines on the post-influenza susceptibility to pneumococcal coinfection. Scientific Reports. 2016 Nov; 6:37045 EP –. https://doi.org/10.1038/srep37045, article.
