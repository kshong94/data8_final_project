#!/usr/bin/env python
# coding: utf-8

# In[2]:


# modules for research report
from datascience import *
import numpy as np
import random
import pandas as pd
import folium
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

# module for YouTube video
from IPython.display import YouTubeVideo

# okpy config
from client.api.notebook import Notebook
ok = Notebook('airbnb-final-project.ok')
_ = ok.auth(inline=True)


# In[ ]:





# # Airbnb Listings and Evictions
# 
# The dataset you will be using is from [Inside Airbnb](http://insideairbnb.com/get-the-data.html), an independent investigatory project that
# collects and hosts substantial Airbnb data on more than 100 cities around the world. The data collected by Inside Airbnb are web-scraped from
# the Airbnb website on a monthly basis. Inside Airbnb was started to investigate the effects of Airbnb on affordable housing and gentrification.
# Its data are made public for free and open for use.  
# 
# We have prepared for you a random subset of Inside Airbnb data from San Francisco collected in June 2020. The data have been
# cleaned for your convenience: all missing values have been removed, and low-quality observations and variables have been filtered
# out. A brief descriptive summary of the dataset is provided below. 
# 
# We are aware that this dataset is potentially significantly larger (in both rows and columns) than other datasets for the project. As a result, 
# you will have many potential directions to conduct your analysis in. At the same time, it is very easy to become overwhelmed or lost with the data.
# We encourage you to reach out by posting your questions on the relevant Piazza thread, or by sending Angela (guanangela@berkeley.edu) or Alan
# (alanliang@berkeley.edu) an email if you need any help.
# 
# **NB: You may not copy any public analyses of this dataset. Doing so will result in a zero.**

# ## Summary
# >Airbnb offers a platform to connect hosts with guests for short-term or long-term lodging accommodations. Compared to similar firms offering vacation rental services
# such as VRBO or HomeAway, Airbnb is the largest and most prominent, with more than 7 million listings worldwide and 2 million people staying in one of its listings
# per night in 2018. Since its founding in 2008, hosts on the platform have served more than 750 million guests, and the firm has grown at an exponential rate globally
# pre-COVID.
# 
# 
# >The data presented are completely from web scraping the Airbnb website in June 2020 for random subset of listings in San Francisco. As a result, the data only contain
# information that a visitor to Airbnb’s site can see. This includes the `listings` table that records all Airbnb units and the `calendar` table that records the
# availabilities for the next 365 days and quoted price per night over the next year of each listing. What each table specifically describes will be gone over in the
# Data Description section below. Note that we do not observe Airbnb transactions or bookings, but only the dates that are available or unavailable through `calendar`.
# 
# 
# >The primary identifier for each listing is the `listing_id` or `id` column (the column name changes depending on the title). Each ID uniquely determines a listing,
# and every listing only has 1 ID. You can visit each listing's URL on Airbnb by going to https://www.airbnb.com/rooms/YOUR_ID_HERE with the id to look up the listing
# on the airbnb website.

# ## Data Description

# The dataset consists of many tables stored in the `data` folder. **You do not have use to all of the tables in your analysis.**
# 1. `listings` provides information on 2000 Airbnb listings in San Francisco. Each row is a unique listing.
# 2. `ratings` contains average ratings for the Airbnb listings across 6 categories and its overall rating. Guests who stay at an Airbnb are eligible to score a listing on each of the categories and on the overall score out of 5.
# 3. `calendar` contains each listing's availability and price over the next year. This data is the same as the calendar that pops up when users try to select the dates of a reservation for a particular listing. For example, the first row means that the listing with ID 40138 was not available on June 8th, 2020. The price per night of this listing is \\$67. 
# 4. `evictions` contains information on evictions in San Francisco, and may be useful if you are interested in determining relationships between Airbnbs and gentrification or evictions.
# 
# 
# There are a lot of columns for many of these datasets, and you probably will only use a few of them. We've selected some of the variables that may be of interest below:
# 
# `listings`:
# * `id`: listing ID.  You can visit each listing's URL on Airbnb by going to https://www.airbnb.com/rooms/YOUR_ID_HERE with the id to look up the listing on the airbnb website.
# * `Name`: listing or rental name
# * `neighborhood` and `neighbourhood_cleansed`: neighborhood of listing
# * `latitude`, `longtitude`: latitude and longitude of listing location. Note that for privacy reasons, this may be approximate.
# * `calculated_host_listings_count`: the number of different listings the host has on Airbnb.
# * `property_type`: type of property the listing is in (e.g. Apartment, Condo, House, etc)
# * `room_type`: type of place (e.g. entire home, private room, etc)
# * `accommodates`: max number of guests
# * `minimum_nights` and `maximum_nights`: minimum and maximum number of nights a reservation can be
# * `availability_X`: availability for the next X days (relative to the scraping date, June 8th, 2020)
# * `amenities`: a list of amenities provided by the listing. Note that each item is an iterable set
# 
# `ratings`: 
# * `review_scores_rating`: review score overall rating of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 100. 
# * `review_scores_accuracy`: review score based on accuracy of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_cleanliness`: review score based on clealiness of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_checkin`: review score based on check-in of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_communication`: review score based on communication with host. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_location`: review score based on location of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# 
# 
# `calendar`:
# * `listing_id`: ID of airbnb listing
# * `date`: date of the potential availability in question
# * `price`: price per night of listing in USD
# * `available`: true or false value representing whether the listing was available.
# 
# `evictions`:
# * `File Date`: date the eviction was reported 
# * `Neighborhood`: neighborhood in which the eviction occurred
# * `Longtitude` and `Latitude`: latitude and longitude of the listing
# * All other columns indicate the reason of the eviction. For example, if an eviction has `True` for the `Non Payment` column and `False` for all other columns, the eviction was due to non-payment. 

# ## Inspiration

# A variety of exploratory analyses, hypothesis tests, and predictions problems can be tackled with this data. Here are a few ideas to get
# you started:
# 
# 
# 1. Can we use Airbnb data to predict which neighborhoods in San Francisco are more gentrified and/or have more evictions?
# 2. Can we predict the overall rating of a listing from one or many of its 6 rating categories? Which of the 6 rating categories best predicts overall rating?  
# 3. Can we predict the average price of a listing (as determined by calendar prices) based on its location, number of bedrooms, or through other features?
# 
# Here are some articles we found to be interesting that may inspire you:
# 1. [Airbnb’s COVID-19 crisis could be a boon for affordable housing](https://www.fastcompany.com/90482662/airbnbs-covid-19-crisis-could-be-a-boon-for-affordable-housing)
# 2. [Identifying salient attributes of peer-to-peer accommodation experience](https://www.tandfonline.com/doi/full/10.1080/10548408.2016.1209153?src=recsys)
# 3. [Airbnb is seeing a surge in demand
# ](https://www.latimes.com/business/story/2020-06-07/airbnb-coronavirus-demand)
# 4. [Airbnb’s Coronavirus Crisis: Burning Cash, Angry Hosts and an Uncertain Future
# ](https://www.wsj.com/articles/airbnbs-coronavirus-crisis-burning-cash-angry-hosts-and-an-uncertain-future-11586365860)
# 5. [Research: When Airbnb Listings in a City Increase, So Do Rent Prices
# ](https://hbr.org/2019/04/research-when-airbnb-listings-in-a-city-increase-so-do-rent-prices)
# 6. [Airbnb is getting ripped apart for asking guests to donate money to hosts
# ](https://www.businessinsider.com/airbnb-asking-renters-to-donate-kindness-cards-backlash-2020-7?utm_source=reddit.com)
# 
# Don't forget to review the [Final Project Guidelines](https://docs.google.com/document/d/1NuHDYTdWGwhPNRov8Y3I8y6R7Rbyf-WDOfQwovD-gmw/edit?usp=sharing) for a complete list of requirements.

# ## Preview
# 
# The tables are loaded in the code cells below. Take some time to explore them!

# In[3]:


# Load the data for airbnb listings
listings = Table().read_table("data/listings.csv")
listings.show(5)


# In[ ]:


# Load in ratings table
ratings = Table().read_table("data/ratings.csv")
ratings.show(5)


# In[ ]:


# Load in the calendar table
calendar = Table().read_table("data/calendar.csv")
calendar.show(5)


# In[ ]:


# Load in the evictions table
evictions = Table().read_table("data/evictions.csv")
evictions.show(5)


# <br>
# 
# # Research Report

# ## Introduction
# 
# *Replace this text with your introduction*

# ## Hypothesis Testing and Prediction Questions
# 
# **Please bold your hypothesis testing and prediction questions.**
# 
# *Replace this text with your hypothesis testing and prediction questions*

# ## Exploratory Data Analysis
# 
# **You may change the order of the plots and tables.**

# **Quantitative Plot:**

# In[ ]:


# Use this cell to generate your quantitative plot


# *Replace this text with an analysis of your plot*

# **Qualitative Plot:**

# In[ ]:


# Use this cell to generate your qualitative plo# Use this cell to generate your qualitative plot


# *Replace this text with an analysis of your plot*

# **Aggregated Data Table:**

# In[ ]:


# Use this cell to generate your aggregated data table


# *Replace this text with an analysis of your plot*

# **Table Requiring a Join Operation:**

# In[ ]:


# Use this cell to join two datasets


# *Replace this text with an analysis of your plot*

# ## Hypothesis Testing
# 
# **Do not copy code from demo notebooks or homeworks! You may split portions of your code into distinct cells. Also, be sure to
# set a random seed so that your results are reproducible.**

# In[ ]:


# set the random seed so that results are reproducible
random.seed(1231)

...


# ## Prediction
# 
# **Be sure to set a random seed so that your results are reproducible.**

# In[ ]:


# set the random seed so that results are reproducible
random.seed(1231)

...


# ## Conclusion
# 
# *Replace this text with your conclusion*

# ## Presentation
# 
# *In this section, you'll need to provide a link to your video presentation. If you've uploaded your presentation to YouTube,
# you can include the URL in the code below. We've provided an example to show you how to do this. Otherwise, provide the link
# in a markdown cell.*
# 
# **Link:** *Replace this text with a link to your video presentation*

# In[ ]:


# Full Link: https://www.youtube.com/watch?v=BKgdDLrSC5s&feature=emb_logo
# Plug in string between "v=" and ""&feature":
YouTubeVideo('BKgdDLrSC5s')


# # Submission
# 
# *Just as with the other assignments in this course, please submit your research notebook to Okpy. We suggest that you
# submit often so that your progress is saved.*

# In[ ]:


# Run this line to submit your work
_ = ok.submit()

