# LOL, assign each (state,trait) pair to one of 6 classes, embed into 6 dimensions, 
#and plot pairplots of the 6 dimensions color coded by (1) state, and then also (2) trait

require(devtools)
install_github('neurodata/lol', build_vignettes=TRUE, force=TRUE)  # install lol with the vignettes
require(lolR)
vignette("lol", package="lolR")  # view one of the basic vignettes

source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")

install.packages('h5')

library(lolR)
library(h5)
library(ggplot2)

setwd("~/JHU/jovo-lab/meditation/data/")
data_path <- './interim/gcca250/'

paths <- list.files(data_path, pattern='.*.h5',full.names=TRUE)
temp <- lapply(paths, h5file)
X <- lapply(temp, function(x) x["latent", "a"][])
rm(temp)

l <- length(X)
X <- lapply(X, as.vector)
X <- as.numeric(unlist(X))
X <- matrix(X ,nrow=l,byrow=TRUE)


Y <- c()
leg <- c()
for(i in seq(length(paths))) {
    path <- paths[i]
    if(grepl('/e_', path)){
        if(grepl('compassion', path)){
            Y <- c(Y, 1)
            leg <- c(leg,'Expert Compassion')
        } else if(grepl('restingstate', path)){
            Y <- c(Y, 2)
            leg <- c(leg,'Expert Resting')
        } else if(grepl('openmonitoring', path)){
            Y <- c(Y, 3)
            leg <- c(leg,'Expert Openmonitoring')
        }
    } else if(grepl('/n_', path)){
        if(grepl('compassion', path)){
            Y <- c(Y, 4)
            leg <- c(leg,'Novice Compassion')
        } else if(grepl('restingstate', path)){
            Y <- c(Y, 5)
            leg <- c(leg,'Novice Resting')
        } else if(grepl('openmonitoring', path)){
            Y <- c(Y, 6)
            leg <- c(leg,'Novice Openmonitoring')
        }
    }
}

r <- 6
result <- lol.project.lol(X, Y, r)
output <- result$Xr
row.names(output) <- leg

pairs(result$Xr, col=alpha(Y, 0.4), pch=19)
legend(x="topright", legend = leg, col=Y, pch=19)

write.table(output, file='./interim/lol_embedding.csv', sep=',',row.names=TRUE, col.names=FALSE)
