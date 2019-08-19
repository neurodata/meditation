library(mgc)
library(gridExtra)
library(grid)

##########################################
## Load Data

setwd("~/JHU/jovo-lab/meditation/data/")
load("./RData/gcca250_decimate_rank3.RData")
data_path <- './interim/gcca_distances/'

distance_files <- list.files(data_path, pattern='.*?_distances.csv',
                             full.names=TRUE)

label_files <- list.files(data_path, pattern='.*?_labels.csv',
                        full.names=TRUE)

names <- c("All Open Monitoring vs. All Compassion",
           "All Resting vs. All Compassion",
           "All Resting vs. All Open Monitoring vs. All Compassion",
           "All Resting vs. All Open Monitoring",
           "Experts All vs. Novices All",
           "Experts Compassion vs. Novice Compassion",
           "Experts Compassion vs. Novices Open Monitoring",
           "Experts Compassion vs. Novices Resting",
           "Experts Meditating vs. Experts Resting",
           "Experts Meditating vs. Novices Resting",
           "Experts Open Monitoring vs. Experts Compassion",
           "Experts Open Monitoring vs. Novices Compassion",
           "Experts Open Monitoring vs. Novices Open Monitoring",
           "Experts Open Monitoring vs. Novices Resting",
           "Experts Resting vs. Experts Compassion",
           "Experts Resting vs. Experts Open Monitoring",
           "Experts Resting vs. Novices Compassion",
           "Experts Resting vs. Novices Open Monitoring",
           "Experts Resting vs. Novices Resting",
           "Novices Meditating vs. Novices Resting",
           "Novices Open Monitoring vs. Novices Compassion",
           "Novices Resting vs. Novices Compassion",
           "Novices Resting vs. Novices Open Monitoring")

select <- c(1)

temp <- lapply(distance_files[select], read.table, sep=',')
dists <- lapply(temp, as.matrix)
temp <- NaN

temp <- lapply(label_files, read.table, sep=',')
labels <- lapply(temp, as.matrix)

##########################################
## Calculate Pvals

get_pvals <- function(results,tol=2) {
    pvals <- as.numeric(unlist(results[seq(3, length(results), 3)]))
    pvals <- signif(pvals,tol)
    return(pvals)
}

# Meditating vs not, all traits
idx <- c(1)
label_files[idx]
results_inter_trait <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

# Inter trait
idx <- c(5)
label_files[idx]
results_inter_trait <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

# intra state, inter state
idx <- c(6,13,19)
names[idx]
results_intra_state_inter_trait <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

# Inter state, pairwise
idx <- c(1,2,4)
label_files[idx]
results_inter_state <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

# Inter state, all
idx <- c(3)
label_files[idx]
results_inter_state3 <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

##
idx <- c(9,10,20)
label_files[idx]
results_meditating_resting <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

## Inter state, intra trait
idx <- c(11,15,16,21,22,23)
label_files[idx]
results_inter_state_intra_trait <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

## Inter state, inter trait
idx <- c(7,8,12,14,17,18)
label_files[idx]
results_inter_state_inter_trait <- mapply(discr.test.one_sample, dists[idx], labels[idx], nperm=1000)

## Big f'n table
# state == mindset
# trait == expert vs. novice
idxs <- c(5, #Inter trait
          1,2,4, #Inter state, pairwise
          3, #Inter state, triplet
          6,13,19, #intra state, inter trait
          9,10,20, #Meditating vs. Resting
          11,15,16,21,22,23, #inter state, intra trait
          7,8,12,14,17,18) #inter state, inter trait
          

results_all <- c(results_inter_trait, #Inter trait
                results_inter_state, #Inter state, pairwise
                results_inter_state3, #inter state, triplet
                results_intra_state_inter_trait, #intra state, inter trait
                results_meditating_resting, #Meditating vs. Resting
                results_inter_state_intra_trait, #inter state, intra trait
                results_inter_state_inter_trait) #inter state, inter trait
                

pvals <- get_pvals(results_all)

ptable <- matrix(pvals ,ncol=1,byrow=FALSE)
colnames(ptable) <- c("GCCA Rank=3")
rownames(ptable) <- names[idxs]
ptable <- as.table(ptable)
names(dimnames(ptable)) <- list('Traits and States', 'Estimated pvals (1000 permuations')


#write.csv(ptable, file='../reports/gcca_pvals_temp.csv')
