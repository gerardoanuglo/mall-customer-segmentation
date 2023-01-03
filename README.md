# mall-customer-segmentation

## Introduction
   As a data scientist for a marketing company, a client has recently acquired a mall in Saint Louis, Missouri and has contacted us to help identify different groups among the customer demographic so the owner can use these segments to accurately plan the marketing strategy. The only data we have is from a previous owner, where they collected certain metrics about the customers.
  
### Objective: Create an unsupervised model using the KMeans algorithm to segment customers into the most optimal number of groups based on key metrics given.  
  
Data Source: [Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Plan of Action
1. Import dependencies
2. Load in files
3. Clean data
4. Explore Data
5. Standardize data 
6. Identify the best number of clusters
7. Build KMeans model
8. Create a column for cluster designation
9. Analyze cluster designation based on key metrics (spending score and income)
10. Understand and describe customer segments. 

### Exploring The Data

Since this dataset was relatively small and organized, there wasn't much cleaning. We checked for duplicate customers, nulls, errors in data values for columns ID, Income, and Spending Score. I changed column names to be clear and concise. Lastly, I checked the data types of all columns to confirm there were no errors. 
With this in mind, I will go straight to the initial data exploration.

To understand the data I used the describe function to identify value ranges and averages for key metrics.

<img width="638" alt="Screenshot 2022-12-01 at 3 04 22 PM" src="https://user-images.githubusercontent.com/85320743/205177612-b544feab-2f95-4317-a7c0-4f657c8b8317.png">

The average age among these customers were 39 years old with a range from 18 to 70 years old. This indicated the dataset didn’t account for anyone younger than 18 because minors aren't the ones paying for goods or services. The average income is 61k with a standard deviation of 26 thousand. The range of income is a minimum of 15k and a maximum of 137k. Finally, the average spending score is 50 out of 100. 

As I mentioned before, I checked for null values and data types.

<img width="513" alt="Screenshot 2022-12-01 at 3 09 08 PM" src="https://user-images.githubusercontent.com/85320743/205178233-83dd0f02-9a58-4036-af7b-bc5dabb9a2bf.png">

Thankfully, we had no null values in our dataset and our data types accurately represented the information in the correct form.

From the dataset we identified annual income and spending score as the two most important metrics to consider.

With this in mind we looked at the correlation between annual income and spending score to identify surface level groupings.

<img width="474" alt="Screenshot 2022-12-01 at 3 48 23 PM" src="https://user-images.githubusercontent.com/85320743/205182954-66552b00-85ca-410e-abbb-51716e2d72f8.png">

From the scatter plot we identified a distinct and condensed cluster in the middle of the graph. They were customers who spent on average the same amount they earned. On the left we have two moderately condensed groups with low spending, one with low income and one with high income.  

At first glance we see 5 groups, but it could be 7 depending on how we consider the farest right points for high and low spenders. These points aren't far enough from the groups to confidently consider them their own groups, yet they aren't close enough to confidently place them in the relatively closest group. We will need to do further analysis to identify the best number of groups.

Identifying the best number of clusters helps avoid plots that look like a single cluster (too few clusters) or look like two clusters are competing for a single densely packed space (too many clusters). 


Consider the distribution of customers by spending groups. The initial dataset came with the column "Spending Score", a metric derived prior to this analysis, but in essence measures one's spending habits. In order to create this visualization, I created another column grouping the customers by spending scores in intervals of 10.

<img width="459" alt="Screenshot 2022-12-01 at 3 50 10 PM" src="https://user-images.githubusercontent.com/85320743/205183179-616b163d-34c0-4475-b15f-d70c8497bd8a.png">

From this initial visualization we concluded the majority of customers had a spending score ranging from 40 to 60. 

Now let's consider the average age of customers by spending groups. Please note this visualization below is zoomed in, the y axis starts at 30 and ends at 46. This is to highlight the distinction that higher spending groups tend to be on average younger compared to low spending groups. 

<img width="473" alt="Screenshot 2022-12-01 at 3 51 04 PM" src="https://user-images.githubusercontent.com/85320743/205183284-3b5d8f37-bd39-4c59-b36a-76cbceac3c5e.png">

We see that on average younger customers spend more money at the mall. Intuitively, while we are young we tend to spend more and as we grow older we tend to be more intentional with our money.

## Standardizing the Data

In order to cluster customers we used the KMeans model which is relatively simple, scales to larger data if needed, and generalizes to clusters of different shapes and sizes.

We standardized the data using SKLearns StandardScaler function to ensure all variables had a similar influence on cluster formation. We assumed the owner wanted us to standardize all data since no instructions were given to leave a key metric out of standardization. This happens at times when a client wants a key metric to hold more influence on cluster formation. 

<img width="820" alt="Screenshot 2022-12-01 at 4 06 26 PM" src="https://user-images.githubusercontent.com/85320743/205185012-9b54fc32-c507-47c2-b7fb-653a922aa673.png">

###  Identifying the Best Number of Clusters

Before the KMeans model was built, we  identified the best number for K using the elbow curve method. We didn’t visually identify K because it is too subjective. We could have used the silhouette score, but it is mathematically rigorous, isn’t subjective, and not intuitive. So in this case we used the Elbow Curve which does involve mathematical calculation, but has some subjectivity built into it and is more intuitive to understand.


We do this is by iterating through KMean models with different K values and taking the inertia (also known as SSE) from each variation of the KMeans model. With the inertia values we plot them with their corresponding k number. The K where the inertia falls suddenly from the previous point and the points following this K only marginally decrease is the most optimal number of clusters to group customers in.

<img width="900" alt="Screenshot 2022-12-01 at 4 14 25 PM" src="https://user-images.githubusercontent.com/85320743/205185876-48962e21-232a-4f11-a1f1-beb8d04d63ee.png">

<img width="473" alt="Screenshot 2022-12-01 at 4 14 08 PM" src="https://user-images.githubusercontent.com/85320743/205185841-d957ab53-a9e7-445e-a1ad-a79d580280e4.png">

Looking at the graph the elbow of the curve is at k = 5, meaning 5 is the most optimal number of clusters to group the customers.

### Building a KMeans Model and Identifying Cluster Designation

We then created a function that builds a KMeans model using our input of k and data. The function returned the data with a new column containing the clusters found. In the table below the cluster designation is called the class.

<img width="638" alt="Screenshot 2022-12-01 at 4 18 58 PM" src="https://user-images.githubusercontent.com/85320743/205186375-75531b48-9f53-46ac-9288-cbdaab8b6262.png">

<img width="350" alt="Screenshot 2022-12-01 at 4 19 34 PM" src="https://user-images.githubusercontent.com/85320743/205186443-e10bb9f1-f2fd-482f-b773-d6fcdeab42ba.png">

To visualize our segmented customers, the class was used as the color for the data points.

<img width="465" alt="Screenshot 2022-12-01 at 4 22 57 PM" src="https://user-images.githubusercontent.com/85320743/205186822-f2c6dbed-1a39-4958-9e3a-4a773d47c59a.png">

The 5 clusters are almost completely distinct from one another. However the central cluster and the upper left cluster have a few points close to each other. 

### Analyzing Cluster Designation Based on Key Metrics.

(spending score and income)

The customers were grouped by their clusters and analyzed by their mean income and spending score to uncover insights about the groups. 

<img width="512" alt="Screenshot 2022-12-01 at 4 27 12 PM" src="https://user-images.githubusercontent.com/85320743/205187311-65cd3402-0ad6-4829-894a-e60d9b43cf12.png">

RESULTS:

0.  Penny Pinchers

Our Penny Pinchers are people who make a lot, but spend a fraction of their income. They tend to be older in established careers with their age averaging around their forties.

1.  Average Joes

Our average joes are around their forties. They earn the average income and spend a little less than what they make. They live within their means, but also will indulge in their interests occasionally.

2.  High Rollers

Our high rollers make a lot, but spend almost as much as they make. Their age is around their early thirties. They love to indulge in their interests. 

3.  Over Spenders

Our over spenders are typically in their twenties. They don't make much, but spend a lot. They don't live within their means, but aspire to one day make enough to afford their lifestyle.

4.  Old Average Joes

Lastly our old average joes, these customers are a lot older and make less than the average person. Nevertheless they still live within their means by spending less than they make. 

### Conclusion

We helped the mall gain a better understanding of its customers to enable better targeting and boost sales by identifying the optimal number of groups and by segmenting customers based on income and spending score. We hope these segments make all the difference and help maximize the ROI of the marketing campaign budget. 


Email: gerardo.angulo20@gmail.com

Linkedin: https://www.linkedin.com/in/gerardo-angulo-7b564218b/


