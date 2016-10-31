library(shiny)
library(googleVis)
library(ggplot2)
library(plotrix)
library(sqldf)
list<-read.csv('pokemonGO.csv')
df<-read.csv('300k.csv')
list$Name<-as.character(list$Name)
list[29,2]<-'Nidoran'
list<-list[-32,]
list$Max.HP<-as.numeric(as.character(list$Max.HP))
c<-as.Date(df$appearedLocalTime)
df$Date<-c
ss<-data.frame(hour=seq(1:24),value=seq(1:24))
clock24.plot(ss$value)

shinyServer(function(input,output){
  output$ui<-renderUI({
    
    tags$img(src = list[list$Name==input$Pkm,6], alt = "photo",height="200", width="200")
    
  })
  DT<-reactive({
  
    id<- list[list$Name==input$Pkm,1]
    table<- df[df$pokemonId==id,]
    table
  })
  output$cp<-renderGvis({
    
     vle<-list[list$Name==input$Pkm,4]
    
    gvisGauge(data.frame(Item='Max CP',Value=vle),options=list(min=0, max=max(list$Max.CP),height=200,width=200))
  })
  
  output$hp<-renderGvis({
    
      vle<-list[list$Name==input$Pkm,5]
    
    gvisGauge(data.frame(Item='Max HP',Value=vle),options=list(min=0, max=max(list$Max.HP),height=200,width=200))
  })
  output$map<-renderPlotly({
    df<- DT()[,c('latitude','longitude')]
    g <- list(
      showland = TRUE,
      landcolor = toRGB("white"),
      subunitcolor = toRGB("white"),
      countrycolor = toRGB("white"),
      countrywidth = 0.5,
      subunitwidth = 0.5
    )
    plot_geo(df, lat = ~latitude, lon = ~longitude) %>%
      add_markers(color='red')%>%
      layout(
        title = paste(input$Pkm,'show up at those locations',sep=' '), geo = g
      )
  })
  output$city<-renderPlotly({
    df<-DT()
    df<-sqldf('SELECT city, count(city) as Frequency from df group by city')
    p <- plot_ly(df, x = ~city, y = ~Frequency, type = 'bar', color = I("black")) %>%
      layout(title = "# Show up among cities")
    p
  })
  output$Gym<-renderPlotly({
    df<-DT()[DT()$gymDistanceKm<5,]
    plot_ly(x=df$gymDistanceKm, type = "histogram",color = I("red"))
    
  })
  output$pop<-renderPlotly({
    df<-DT()
    plot_ly(x=df$population_density, type = "histogram",color = I("blue"))
    
  })
  output$hour<-renderPlot({
    tmp<-DT()
    tmp<-sqldf('SELECT appearedHour as Hour, count(appearedHour) as Count from tmp group by Hour order by Hour')
    clock24.plot(tmp$Count,seq(0:23),labels=0:23,line.col='red',lwd=3,mar=c(2,2,4,2),main=paste(input$Pkm,'in 24 hours',sep=' '))
    
  })
  
  output$minute<-renderPlotly({
    tmp<-DT()
    tmp<-sqldf('SELECT appearedMinute as Minute, count(appearedMinute) as Count from tmp group by Minute order by Minute')
    
    plot_ly(tmp,x=~Minute,y=~Count,color = I('red'),type='bar')%>%
      layout(title = "Show up Count in Minutes",
             scene = list(
               xaxis = list(title = "Minutes"), 
               yaxis = list(title = "Count")))
  })
  
  output$hm<-renderPlotly({
    tmp<-DT()
    tmp<-sqldf('SELECT appearedHour as Hour,appearedMinute as Minute, count(appearedMinute) as Count from tmp group by Hour,Minute order by Hour,Minute')
    plot_ly(tmp,x=~Minute,y=~Hour,z=~Count,colorscale = "Greys", type = "heatmap")
  })
  output$pie<-renderPlotly({
    tmp<-DT()
    tmp<-sqldf('SELECT appearedTimeOfDay as Period, count(appearedTimeOfDay) as Count from tmp group by Period')
    plot_ly(tmp, labels = ~Period, values = ~Count, type = 'pie') %>%
      layout(title = 'Proportion of showing up period',
             xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
             yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
    
  })
  output$pie2<-renderPlotly({
    
    tmp<-DT()
    tmp<-sqldf('SELECT weather, count(weather) as Count from tmp group by weather')
    plot_ly(tmp, labels = ~weather, values = ~Count, type = 'pie') %>%
      layout(title = 'Proportion of showing up under specific weathers',
             xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
             yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
    
    
  })
  
  output$tp1<-renderPlotly({
    tmp<-DT()
    tmp<-sqldf('SELECT temperature, pressure, count(pressure) as Count from tmp group by temperature, pressure')
    plot_ly(tmp,x=~temperature,y=~pressure,z =~Count,color = ~Count)%>%
      layout(title = 'Showing up occurence vs temperature and pressure',
             scene = list(
               xaxis = list(title = "Temperature"), 
               yaxis = list(title = "Pressure")))
  })
  
  output$wd2<-renderPlotly({
    tmp<-DT()
    tmp<-sqldf('SELECT windSpeed, windBearing, count(windBearing) as Count from tmp group by windSpeed, windBearing')
    plot_ly(tmp,x=~windSpeed,y=~windBearing,z =~Count,color = ~Count)%>%
      layout(title = 'Showing up occurence vs wind statistics',
             scene = list(
               xaxis = list(title = "windSpeed"), 
               yaxis = list(title = "windBearing")))
  })
  
  
})