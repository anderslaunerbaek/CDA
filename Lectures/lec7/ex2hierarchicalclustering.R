rm(list=ls())
library(dendextend) #easy visualization manipulation and comparison of dendrograms 
library(circlize) #circle dendograms

class <- as.factor(as.matrix(read.csv("~/DTU/Courses/CDA/Lectures/lec7/data/ziplabel.csv")))
X <- read.csv("~/DTU/Courses/CDA/Lectures/lec7/data/zipdata.csv")
N <- dim(X)[1]
p <- dim(X)[2]


# draw random samples
Xsample <- as.matrix(X[sample(nrow(X), size = 9, replace = F), ])
rotate <- function(x) t(apply(x, 2, rev))
par(mfrow=c(3, 3)) 
for(i in 1:9){
  M <- matrix(as.numeric(Xsample[i, ]), sqrt(p), byrow = TRUE)
  w <- apply(X, 1, function(x, want) isTRUE(all.equal(x, want)), Xsample[i, ])
  image(rotate(M), col = grey.colors(255), main = paste("Class:", class[w]))}

# sample-sample distance method=c("euclidean", "maximum", "manhattan", "canberra", "binary" ,"minkowski")
d <- dist(as.matrix(X), "euclidean")   # find distance/dissimilarity matrix 

# group-group distance\linkage , agglomeration method
# dendlist of all possible methods
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty",
                 "median", "centroid", "ward.D2")
ZIP_dendlist <- dendlist()
for(i in seq_along(hclust_methods)) {
    hc <- hclust(d, method = hclust_methods[i])
    ZIP_dendlist <- dendlist(ZIP_dendlist, hc)
}
names(ZIP_dendlist) <- hclust_methods
# ZIP_dendlist

######################################################################
#Change parameters
L <- 8 # linkage
print(hclust_methods[L])
K <- 9
#number of clusters
##########################################
hc <- ZIP_dendlist[[L]]
par(mfrow=c(1,1)) 
plot(hc,main =paste("Hierarchical clusterering","\n"," with linkage:",hclust_methods[L]))
# Cut tree into 10 groups
grp <- cutree(as.hclust(hc), k = K)
# Number of members in each cluster
# Get the class for the members of cluster 1
class[grp == 1]
rect.hclust(as.hclust(hc), k = K, border = rainbow(K))

dend <- as.dendrogram(hc)
# rotate the dendogram
dend <- rotate(dend, 1:N)

# Color the branches based on the clusters:
dend <- color_branches(dend, k=K,col=rainbow(K)) #, groupLabels=

# Manually match the labels with colors
labels_colors(dend) <- rainbow(K)[sort_levels_values(as.numeric(class)[order.dendrogram(dend)])]
labels(dend) <- paste(as.character(class)[order.dendrogram(dend)], sep = "")
dend <- set(dend, "labels_cex", 0.7)

#plot
par(mar = c(3,3,3,7))
plot(dend, main = "Clustered ZIP data set", horiz =  TRUE,  nodePar = list(cex = .007))
numbers <- rev(levels(class))
legend("topleft", legend = paste("cluster", numbers), fill = rev(rainbow(K)))

par(mar = rep(0,4))
c <- circlize_dendrogram(dend)
legend("topleft", legend = paste("cluster", numbers), fill = rev(rainbow(K)))

