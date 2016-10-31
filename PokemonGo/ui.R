library(shiny)
library(plotly)
library(sqldf)
library(shinythemes)

list<-read.csv('pokemonGO.csv')
list$Name<-as.character(list$Name)
list[29,2]<-'Nidoran'
list<-list[-32,]
shinyUI(fluidPage(theme = shinytheme("slate"),
                  titlePanel('Pokemon Go EDA'),
                  sidebarLayout(
                    sidebarPanel(
                      uiOutput("ui",align="center"),
                      br(),
                      br(),
                      selectInput('Pkm',"Pokemon",choices =as.character(list$Name),selected = as.character(list$Name)[1]),
                      br(),
                      h4('Author: Jason Liu'),
                      a(h5("LinkedIn"),href="https://nl.linkedin.com/in/jiashen-liu-4658aa112",target="_blank"),
                      a(h5("Github"),href="https://github.com/liujiashen9307/",target="_blank"),
                      br(),
                      h4('Claim: This Shiny App uses the data set from Kaggle.com. Packages used in this app are: Shiny, plotly and sqldf'),
                      a(h5("Data Link 1"),href="https://www.kaggle.com/semioniy/predictemall",target="_blank"),
                      a(h5("Data Link 2"),href="https://www.kaggle.com/abcsds/pokemongo",target="_blank")
                      
                      
                    ),
                    mainPanel(
                      tabsetPanel(
                        tabPanel('Geo Data',
                                 p(h3('Statistics of Pokemon')),
                                 splitLayout(cellwidths=c("50%","50%"),htmlOutput("cp"),htmlOutput("hp")),
                        p(h3('Map of Pokemon')),
                        plotlyOutput('map'),
                        p(h3('Frequency of showing up among cities')),
                        plotlyOutput('city'),
                        p(h3('Pokemon Gym Distance Distribution')),
                        plotlyOutput('Gym'),
                        p(h3('Population Density Distribution')),
                        plotlyOutput('pop')),
                        tabPanel('Time Data',
                        p(h3('Show up data Heatmap')),
                        plotlyOutput('hm'),
                        p(h3('Show up data by Hour')),
                        plotOutput('hour'),
                        p(h3('Show up data by Minute')),
                        plotlyOutput('minute'),
                        p(h3('Proportion of showing up periods')),
                        plotlyOutput('pie')
                        
                            ),
                        tabPanel('Weather Data',
                                 p(h3('Proportion of showing up under different Weathers')),
                                 plotlyOutput('pie2'),
                                 p(h3('Temperature and Pressure vs Showing up Count')),
                                 plotlyOutput('tp1'),
                                 p(h3('Wind Effects vs Showing up Count')),
                                 plotlyOutput('wd2')
                                 
                                 )
                      )
                      
                      
                    )
                  )
                  
                  ))