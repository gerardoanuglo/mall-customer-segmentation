# mall-customer-segmentation

## Introduction
  As a data scientist for a marketing company, a new client has recently accquired a mall in Ames, Iowa. The owner has contacted us to help identify different groups among the customer demographic. The owner wants to understand the customers so the segments can be given to the marketing team for planning the strategy accurately. The only data we have is from previous owner, were they collected certain metrics about the customers.
  
### Objective: Create a unsupervised model using the KMeans algorithm to segment customers into the most optimal number of groups based on key metrics given.  
  
Data Source: [Kaggle Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)

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

Since this dataset was relatively small and organized, there wasn't much cleaning. We checked for duplicate customers, nulls, errors in data values for columns ID, Income, and Spending Score. I changed column names to be clear and concise. Lastly I checked the data types of all columns to confirm there was no errors. 
With this in mind, I will go straight to the initial data exploration.

First, I wanted to get a understanding of the data. I used the describe function to identify value ranges and averages for key metrics.

<img width="638" alt="Screenshot 2022-12-01 at 3 04 22 PM" src="https://user-images.githubusercontent.com/85320743/205177612-b544feab-2f95-4317-a7c0-4f657c8b8317.png">

The average age among these customers is 39 years old with a range from 18 to 70 years old. This indicates the dataset does not account for anyone younger than 18 because minors aren't the ones paying of goods or services. The average income is 61k with a standard deviation of 26 thousand. The range of income is a minimum of 15k and a max of 137k. Finally, the average spending score is 50 out of 100. 

As I mentioned before, I checked for null values and data types.

<img width="513" alt="Screenshot 2022-12-01 at 3 09 08 PM" src="https://user-images.githubusercontent.com/85320743/205178233-83dd0f02-9a58-4036-af7b-bc5dabb9a2bf.png">

Thankfully, we have no null values in our dataset and our data types accurately represent the information in the correct form.

From the dataset we identified annual income and spending score as the two most important metrics to consider.

With this in mind we will take a look at the correlation between annual income and spending score to identify surface level/ initial groupings.

<img width="474" alt="Screenshot 2022-12-01 at 3 48 23 PM" src="https://user-images.githubusercontent.com/85320743/205182954-66552b00-85ca-410e-abbb-51716e2d72f8.png">

A few things we can take away from this scatter plot. We can distinctly indentify a condesed cluster in the middle of the graph, customers who spend on average the same amount they earn. On the left we have two moderatley condensed groups with low spending, one with low income and one with high income.  

At first glance we see 5 groups, but it could be 7 depending on how we consider the farest right points for high and low spenders. These points aren't far enough from the groups to confidently consider them their own groups, yet they aren't close enough to confidently place in them in the relatively closest group. We will need to do further analysis to identify the best number of groups.

Identifing the best number of clusters will help us avoid plots that look like a single cluster (too few clusters) or look like two clusters are competing for a single densely packed space (too many clusters). 

Lets continue with the data exploration.

Consider the distribution of customers by spending groups. The initial dataset came with the column "Spending Score", a metric derieved prior to this analysis, but in essence measures ones spending habits. In order to create this visualization, I created another column grouping the customers by spending score in intervals of 10.

<img width="459" alt="Screenshot 2022-12-01 at 3 50 10 PM" src="https://user-images.githubusercontent.com/85320743/205183179-616b163d-34c0-4475-b15f-d70c8497bd8a.png">

From this initial visualization we can conclude the majority of customers have a spending score ranging from 40 to 60. Other then this insight, there is no clear distinction of customer groupings here.

Now lets consider the average age of customers by spending groups. Please note this visulization below is zoomed in, the y axis starts at 30 and ends at 46. This is to highlight the slight distinction that higher spending groups tend to be on average younger compared to low spending groups. 

<img width="473" alt="Screenshot 2022-12-01 at 3 51 04 PM" src="https://user-images.githubusercontent.com/85320743/205183284-3b5d8f37-bd39-4c59-b36a-76cbceac3c5e.png">

We can see that on average younger customers are spending more at the mall. Intuitively, while we are young we tend to spend more and as we grow older we tend to be more methodical with our money.

## Standardizing the Data

Since we are trying to cluster customers using the KMeans model we will need to standardize the data. This model is relatively simple, scales to larger data if needed, and generalizes to clusters of different shapes and sizes.

We will standardize data using SKLearns StandardScaler function to ensure all variables hava a similar influence on cluster formation. We will assume the owner wants us to standardize all data since no instructions were given to leave a key metric out of standardization. This happens at times when a client wants a key metric to hold more influence on cluster formation. 

<img width="820" alt="Screenshot 2022-12-01 at 4 06 26 PM" src="https://user-images.githubusercontent.com/85320743/205185012-9b54fc32-c507-47c2-b7fb-653a922aa673.png">

###  Identifing the Best Number of Clusters

Before we can build a KMeans model, we will need to identify the best number for K using the elbow curve method. We won't be visually identifing K because it is to subjective. We could use the silhouett score, but this method is less intuitive when representing the process. 
The Elbow Curve helps select the optimal number of clusters for KMeans clustering. How we do this is by iterating through KMean models with different K values and taking the inertia (also known as SSE) from each variation of the KMeans model. With the inertia values we plot them with their corresponding k number. The K where the inertia falls suddenly from the previous point and the points following this K only marginally decrease is the most optimal number of clusters to group customers in.

<img width="900" alt="Screenshot 2022-12-01 at 4 14 25 PM" src="https://user-images.githubusercontent.com/85320743/205185876-48962e21-232a-4f11-a1f1-beb8d04d63ee.png">

<img width="473" alt="Screenshot 2022-12-01 at 4 14 08 PM" src="https://user-images.githubusercontent.com/85320743/205185841-d957ab53-a9e7-445e-a1ad-a79d580280e4.png">

Looking at the graph the elbow of the curve is at k = 5, meaning 5 is the most optimal number of clusters to group the customers.

### Building a KMeans Model and Identifing Cluster Designation

We will create a function that builds a KMeans model using our input of k and data. The function will return the data with a new column containing the clusters found. In the table below the cluster designation is called the class.

<img width="638" alt="Screenshot 2022-12-01 at 4 18 58 PM" src="https://user-images.githubusercontent.com/85320743/205186375-75531b48-9f53-46ac-9288-cbdaab8b6262.png">

<img width="350" alt="Screenshot 2022-12-01 at 4 19 34 PM" src="https://user-images.githubusercontent.com/85320743/205186443-e10bb9f1-f2fd-482f-b773-d6fcdeab42ba.png">

Now its time to visualize our segmented customers, using the class as the color for the data points.

<img width="465" alt="Screenshot 2022-12-01 at 4 22 57 PM" src="https://user-images.githubusercontent.com/85320743/205186822-f2c6dbed-1a39-4958-9e3a-4a773d47c59a.png">

The 5 clusters are almost completely distinct from one another. However the central cluster and the upper left cluster have a few points close to each other. 

### Analyzing Cluster Designation Based on Key Metrics.

(spending score and income)

We will group the customers by there clusters and anlayze the mean income and spend score to uncover insights about the groups. 

<img width="512" alt="Screenshot 2022-12-01 at 4 27 12 PM" src="https://user-images.githubusercontent.com/85320743/205187311-65cd3402-0ad6-4829-894a-e60d9b43cf12.png">

Here are the mall customer segments. 

0.  Penny Pinchers

Our Penny Pinchers, are people who make a lot, but spend a friction of their income. They tend to be older in established careers with their age averaging around their fourties.

1.  Average Joes

Our average joes are around their fourties. They earn the average income and spend a little less then what they make. They live within their means, but also will indulge in their interests occasionally.

2.  High Rollers

Our high rollers make a lot, but spend almost as much as they make. Their age is around early thirites. They love to indulge in their interests. 

3.  Over Spenders

Our over spenders are typically in their twenties. They don't make much, but spend a lot. They don't live within their means, but aspire to one day make enough to afford their lifestyle.

4.  Old Average Joes

Lastly our old average joes, these customers are a lot older and make less then the average person. Nevertheless they still live within their means by spending less then they make. 

### Conclusion

We helped the mall gain a better understanding of its customers to enable better targeting and boost sales by identifying the optimial number of groups and by segmenting customers based on income and spending score. We hope these segments make all the difference and help maximize the ROI of the maketing campaign budget. 


Email: gerardo.angulo20@gmail.com

Linkedin: https://www.linkedin.com/in/gerardo-angulo-7b564218b/
