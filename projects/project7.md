# Projet : Construction d'une application R Shiny pour les prévisions de ventes de produits pharmaceutiques

## Description

Dans ce projet, vous allez apprendre à construire une application interactive R Shiny pour effectuer des prévisions de ventes de produits pharmaceutiques. Cette application permet aux utilisateurs d'analyser les données de ventes historiques et de prédire les ventes futures en utilisant des modèles de prévision avancés. Elle est particulièrement utile pour les gestionnaires de produits et les analystes de données dans l'industrie pharmaceutique qui cherchent à optimiser les stocks et à anticiper les demandes de marché.

## Image
imgs/project7/project7.png

## Instructions

L'application que vous allez construire dispose de plusieurs fonctionnalités, telles que, la visualisation des tendances des ventes, l'ajustement des modèles de prévision et l'affichage des prévisions. Vous pouvez accéder à une démonstration de cette application à l'adresse suivante : https://youtu.be/-WUmn-ctRx8 

L'application que vous devez construire doit ressembler à celle disponible ici : [Forecaster](https://josueafouda.shinyapps.io/Forecaster/).

1. **Téléchargement de l'ensemble des données [productdb.rds](https://drive.google.com/file/d/1zqrOnlGrVUvxhOnP_Bkj6NtZOn2779uE/view?usp=sharing)**

2. **Installation des Pré-requis** :

    - Assurez-vous que R et RStudio sont installés sur votre machine.

    - Installez le package `shiny`.

    - Installez les autres packages nécessaires, tels que `ggplot2`, `plotly`, `tidyr`, `DT`, `TSstudio`, `lubridate`, `prophet`, `shinythemes`, `forecast`, et `dplyr`.

3. **Créez un projet RStudio** 

4. **Développement de l'Interface Utilisateur (UI)** :
    - Créez un fichier `ui.R` pour définir l'interface utilisateur de l'application.
    - Utilisez des éléments d'interface comme les `sidebarPanel`, `mainPanel`, `plotOutput`, et `tableOutput` pour construire votre interface.

5. **Développement de la Logique Serveur (Server)** :
    - Créez un fichier `server.R` pour définir la logique serveur de l'application.
    - Importez les données, ajustez le modèle de prévision et générez les graphiques et les tables de prévision.

6. **Lancement de l'Application** :
    - Assurez-vous que vos fichiers `ui.R` et `server.R` sont dans le même répertoire.
    - Lancez l'application en utilisant la commande suivante dans RStudio :
      ```R
      shinyApp(ui, server)
      ```

7. **Déploiement sur shinyapps.io** :
    - Créez un compte sur [shinyapps.io](https://www.shinyapps.io/).
    - Suivez les instructions pour déployer votre application depuis RStudio :

8. **Documentation et Présentation** :
    - Documentez votre code et créez une présentation de votre application.
    - Vous pouvez réaliser une vidéo de présentation pour montrer les fonctionnalités clés et expliquer comment utiliser votre application.

Ce projet vous aidera à développer des compétences pratiques en R Shiny et en prévision de séries temporelles, tout en vous offrant une application concrète que vous pouvez utiliser ou présenter à des employeurs potentiels.


## Resources
- [Ensemble de donnée productdb.rds](https://drive.google.com/file/d/1zqrOnlGrVUvxhOnP_Bkj6NtZOn2779uE/view?usp=sharing)
- [Vidéo de présentation de l'application](https://youtu.be/-WUmn-ctRx8?si=dtMX9ljBBmQVNcsQ)
- [Lien de l'application à reproduire](https://josueafouda.shinyapps.io/Forecaster/)
- [Développement Web en Data Science avec R Shiny sans HTML, CSS, PHP ni JavaScript](https://www.amazon.fr/dp/B095Q5HCTW?ref_=ast_author_ofdp)
- [Formation R Shiny pour les débutants](https://youtu.be/4XGI_ye0y4M)
- [Documentation R Shiny](https://shiny.posit.co/r/getstarted/shiny-basics/lesson1/index.html)
- [Apprendre à programmer avec R et RStudio](https://www.youtube.com/playlist?list=PLmJWMf9F8euQFQSvMSnFiEKxIuAuSFtXt)
- [Comment déployer une application web R Shiny sur shinyapps.io](https://youtu.be/56Lo1oNqpCw)


## Execution du Projet

🖥 **Interface Utilisateur (UI) : Script ui.R**

```R
library(shiny)
library(dplyr)
library(plotly)
library(tidyr)
library(DT)
library(TSstudio)
library(lubridate)
library(forecast)
library(prophet)
library(shinythemes)

# Define UI for application that draws a histogram
fluidPage(
  
  # Thème
  # themeSelector(),
  theme = shinytheme("cyborg"),
  
  # Pour centrer les textes de h1 et h4
  tags$head(
    tags$style(HTML("
      h1, h3 {
        text-align: center;
      }
    "))
  ),
  
  # Titre de l'application
  h1("Forecaster Web App", style = "font-family: 'Jura'; 
     color: red; font-size: 80px;"),
  h3(tags$b("R Shiny web application for forecasting the Sales Revenue or the Quantity of Products in a pharmacy.")),
  br(),
  h4(tags$a("Author : Josué AFOUDA", 
            href = 'https://www.linkedin.com/in/josu%C3%A9-afouda/')),
  
  h4(tags$a("Learn R Shiny",href='https://youtu.be/4XGI_ye0y4M?si=_i7Zcpg91s8XavfU')),
  
  br(),
  
  fluidRow(column(12, wellPanel(
    radioButtons(
      "forecastmetric",
      label = "Forecasting Metric",
      choices = c("Sales Revenue", "Quantity"),
      inline = TRUE
    )
  ))), 
  
  fluidRow(column(2, wellPanel(
    selectInput("categ", 
                label = "Categorie Name", 
                choices = unique(productdb$CHEMSUB))
  )),
  column(10, wellPanel(plotlyOutput("top5plot")))), 
  
  # fluidRow(column(12, wellPanel(uiOutput('ProductControl')))),
  
  fluidRow(column(
    2,
    wellPanel(
      selectInput("prods", 
                  label = "Product Name", 
                  choices = unique(productdb$BNFNAME)),
      br(),
      checkboxInput("decompose", label = "Decompose ETS", value = TRUE),
      br(),
      selectInput(
        "forecastmodel",
        label = "Forecasting Model",
        choices = c(
          "ETS" = "auto",
          "Holt-Winters" = "hw",
          "TBATS" =
            "tbats",
          "Auto ARIMA" = "autoarima",
          "Facebook Prophet" =
            "fbpro"
        )
      )
    )
  ),
  column(5, plotlyOutput("actual_ts")),
  column(5, plotOutput("autoplotforecast"))), 
  
  fluidRow(column(12, 
                  numericInput(
                    "horizon", 
                    label = "Horizon of Prevision",
                    value = 6,
                    min = 1
                  ),
                  DTOutput("forecastdata")))
  
)
```

Le code `ui.R` définit l'interface utilisateur (UI) d'une application R Shiny pour la prévision des ventes de produits pharmaceutiques. Voici une explication détaillée pour aider les débutants à comprendre chaque partie du code :

- Importation des bibliothèques

```r
library(shiny)
library(dplyr)
library(plotly)
library(tidyr)
library(DT)
library(TSstudio)
library(lubridate)
library(forecast)
library(prophet)
library(shinythemes)
```

Ces lignes importent les bibliothèques nécessaires pour construire l'application. Chaque bibliothèque a des fonctionnalités spécifiques :
- `shiny`: pour créer des applications web interactives.
- `dplyr`, `tidyr`, `lubridate`: pour manipuler les données.
- `plotly`: pour créer des graphiques interactifs.
- `DT`: pour créer des tableaux interactifs.
- `TSstudio`, `forecast`, `prophet`: pour l'analyse et la prévision des séries temporelles.
- `shinythemes`: pour ajouter des thèmes à l'application Shiny.

- Définition de l'interface utilisateur

```r
fluidPage(
  
  # Thème
  theme = shinytheme("cyborg"),
  
  # Pour centrer les textes de h1 et h4
  tags$head(
    tags$style(HTML("
      h1, h3 {
        text-align: center;
      }
    "))
  ),
  
  # Titre de l'application
  h1("Forecaster Web App", style = "font-family: 'Jura'; 
     color: red; font-size: 80px;"),
  h3(tags$b("R Shiny web application for forecasting the Sales Revenue or the Quantity of Products in a pharmacy.")),
  br(),
  h4(tags$a("Author : Josué AFOUDA", 
            href = 'https://www.linkedin.com/in/josu%C3%A9-afouda/')),
  h4(tags$a("Learn R Shiny",href='https://youtu.be/4XGI_ye0y4M?si=_i7Zcpg91s8XavfU')),
  
  br(),
  
  fluidRow(column(12, wellPanel(
    radioButtons(
      "forecastmetric",
      label = "Forecasting Metric",
      choices = c("Sales Revenue", "Quantity"),
      inline = TRUE
    )
  ))), 
  
  fluidRow(column(2, wellPanel(
    selectInput("categ", 
                label = "Categorie Name", 
                choices = unique(productdb$CHEMSUB))
  )),
  column(10, wellPanel(plotlyOutput("top5plot")))), 
  
  fluidRow(column(
    2,
    wellPanel(
      selectInput("prods", 
                  label = "Product Name", 
                  choices = unique(productdb$BNFNAME)),
      br(),
      checkboxInput("decompose", label = "Decompose ETS", value = TRUE),
      br(),
      selectInput(
        "forecastmodel",
        label = "Forecasting Model",
        choices = c(
          "ETS" = "auto",
          "Holt-Winters" = "hw",
          "TBATS" = "tbats",
          "Auto ARIMA" = "autoarima",
          "Facebook Prophet" = "fbpro"
        )
      )
    )
  ),
  column(5, plotlyOutput("actual_ts")),
  column(5, plotOutput("autoplotforecast"))), 
  
  fluidRow(column(12, 
                  numericInput(
                    "horizon", 
                    label = "Horizon of Prevision",
                    value = 6,
                    min = 1
                  ),
                  DTOutput("forecastdata")))
  
)
```

- Structure de l'interface

    - `fluidPage`
La fonction `fluidPage` crée une page web avec un design fluide qui s'adapte à la taille de l'écran.

    - Thème
```r
theme = shinytheme("cyborg"),
```
Cette ligne applique un thème visuel à l'application en utilisant le thème "cyborg" de `shinythemes`.

    - Style des titres
```r
tags$head(
  tags$style(HTML("
    h1, h3 {
      text-align: center;
    }
  "))
)
```
Ces lignes ajoutent du CSS pour centrer les titres `<h1>` et `<h3>` sur la page.

    - Titre de l'application et description
```r
h1("Forecaster Web App", style = "font-family: 'Jura'; 
   color: red; font-size: 80px;"),
h3(tags$b("R Shiny web application for forecasting the Sales Revenue or the Quantity of Products in a pharmacy.")),
br(),
h4(tags$a("Author : Josué AFOUDA", 
          href = 'https://www.linkedin.com/in/josu%C3%A9-afouda/')),
h4(tags$a("Learn R Shiny",href='https://youtu.be/4XGI_ye0y4M?si=_i7Zcpg91s8XavfU')),
```
Ces lignes définissent le titre principal de l'application, une brève description, et des liens vers l'auteur et un tutoriel.

- Sélection de la métrique de prévision
```r
fluidRow(column(12, wellPanel(
  radioButtons(
    "forecastmetric",
    label = "Forecasting Metric",
    choices = c("Sales Revenue", "Quantity"),
    inline = TRUE
  )
))),
```
Cette section ajoute un groupe de boutons radio pour choisir entre la prévision du revenu des ventes ou de la quantité de produits.

- Sélection de la catégorie et graphique des top 5 produits
```r
fluidRow(column(2, wellPanel(
  selectInput("categ", 
              label = "Categorie Name", 
              choices = unique(productdb$CHEMSUB))
)),
column(10, wellPanel(plotlyOutput("top5plot")))),
```
Ces lignes créent une sélection déroulante pour choisir une catégorie de produits et un graphique interactif pour afficher les cinq produits les plus vendus de cette catégorie.

- Sélection du produit et des paramètres de prévision
```r
fluidRow(column(
  2,
  wellPanel(
    selectInput("prods", 
                label = "Product Name", 
                choices = unique(productdb$BNFNAME)),
    br(),
    checkboxInput("decompose", label = "Decompose ETS", value = TRUE),
    br(),
    selectInput(
      "forecastmodel",
      label = "Forecasting Model",
      choices = c(
        "ETS" = "auto",
        "Holt-Winters" = "hw",
        "TBATS" = "tbats",
        "Auto ARIMA" = "autoarima",
        "Facebook Prophet" = "fbpro"
      )
    )
  )
),
column(5, plotlyOutput("actual_ts")),
column(5, plotOutput("autoplotforecast"))),
```
Cette section permet de sélectionner un produit, choisir d'afficher la décomposition de la série temporelle, et sélectionner un modèle de prévision. Les graphiques interactifs et statiques affichent les séries temporelles actuelles et les prévisions respectivement.

- Paramètre d'horizon de prévision et affichage des données prévisionnelles
```r
fluidRow(column(12, 
                numericInput(
                  "horizon", 
                  label = "Horizon of Prevision",
                  value = 6,
                  min = 1
                ),
                DTOutput("forecastdata")))
```
Cette section ajoute un champ numérique pour définir l'horizon de prévision (en mois) et un tableau interactif pour afficher les données prévisionnelles.

- Conclusion :

Ce code `ui.R` configure l'interface utilisateur pour une application R Shiny qui permet de sélectionner des produits pharmaceutiques, choisir des modèles de prévision, et visualiser les résultats sous forme de graphiques interactifs et de tableaux. Les utilisateurs peuvent interagir avec les contrôles pour explorer différentes prévisions de vente ou de quantité de produits.


🖥 **Logique Serveur (backend) : Script server.R**

```R
library(shiny)
library(dplyr)
library(plotly)
library(tidyr)
library(DT)
library(TSstudio)
library(lubridate)
library(forecast)
library(prophet)
library(shinythemes)

# Importation des données
productdb <- readRDS("productdb.rds")

# Médicaments les plus prescrits en terme de quantité
productdb2 <- productdb %>%
  group_by(BNFNAME, CHEMSUB) %>%
  summarise(SUMQ = sum(QUANTITY)) %>%
  arrange(desc(SUMQ))

# Fonction pour trouver les 5 premiers médicaments
  # (en terme de qté prescrite) dans une même catégorie
getTop5 <- function(x) {
  # x : catégorie du produit
  result <- productdb2 %>%
    filter(CHEMSUB == x) %>%
    head(5) %>%
    pull(BNFNAME)
  
  return(result)
}

# Define server logic required to draw a histogram
function(input, output, session) {
  
  # Dataframe réactive où se trouve les 5 meilleurs produits dans une même catégorie
  getSalesData <- reactive({
    if(input$forecastmetric == "Quantity") {
      productdb %>%
        filter(BNFNAME %in% getTop5(input$categ)) %>%
        select(month, BNFNAME, QUANTITY) %>%
        rename(Month = month, Product = BNFNAME, Metric = QUANTITY)
    } else if(input$forecastmetric == "Sales Revenue") {
      productdb %>%
        filter(BNFNAME %in% getTop5(input$categ)) %>%
        select(month, BNFNAME, ACTCOST) %>%
        rename(Month = month, Product = BNFNAME, Metric = ACTCOST)
    }
  })
  
  # Graphique montrant les 5 meilleurs produits dans une même catégorie
  output$top5plot <- renderPlotly({
    
    plot_ly(
      getSalesData(),
      x = ~ Month,
      y = ~ Metric,
      color = ~ Product,
      type = 'scatter',
      mode = 'lines',
      text = ~ paste("Product: ", Product)
    ) %>%
      layout(title = paste("Top 5 Products in the ", input$categ, "Category"))
    
  })
  
  # Dataframe filtrée du produit(médicament) sélectionné par l'utilisateur
  getProdData <- reactive({
    if(input$forecastmetric == "Quantity") {
      productdb %>%
        filter(BNFNAME == input$prods) %>%
        mutate(Date = ymd(paste(month, "20", sep = "-"))) %>%
        select(month, QUANTITY, Date) %>%
        rename(Metric = QUANTITY)
    } else if(input$forecastmetric == "Sales Revenue") {
      productdb %>%
        filter(BNFNAME == input$prods) %>%
        mutate(Date = ymd(paste(month, "20", sep = "-"))) %>%
        select(month, ACTCOST, Date) %>%
        rename(Metric = ACTCOST)
    }
  })
  
  # Création d'un objet ts (Série temporelle du Produit sélectionné par l'utilisateur)
  ts_data <- reactive({
    ts(
      data = getProdData()$Metric,
      start = c(year(min(getProdData()$Date)), month(min(getProdData()$Date))),
      frequency = 12
    )
  })
  
  # Affichage de la série temporelle du Produit sélectionné par l'utilisateur
  output$actual_ts <- renderPlotly({
    if(input$decompose) {
      ts_decompose(ts_data())
    } else {
      ts_plot(ts_data(), title = input$prods)
    }
  })
  
  #Dataframe pour la prévision avec Facebook prophet
  prophet_df <- reactive({
    getProdData() %>%
      select(Date, Metric) %>%
      rename(ds = Date, y = Metric)
  })
  
  # Modèle Prophet
  md1 <- reactive({
    prophet(prophet_df())
  })
  
  # Modèle Auto Arima
  md2 <- reactive({
    forecast(auto.arima(ts_data()))
  })
  
  # Modèle TBATS
  md3 <- reactive({
    forecast(tbats(ts_data()))
  })
  
  # Modèle ETS
  md4 <- reactive({
    forecast(ts_data())
  })
  
  # Modèle Holt-Winters
  md5 <- reactive({
    forecast(HoltWinters(ts_data()))
  })
  
  # Affichage des prévisions dans un graphique
  output$autoplotforecast <- renderPlot({
    # "Auto","Holt-Winters","TBATS","Auto ARIMA"
    if (input$forecastmodel == "fbpro"){
      plot(
        md1(),
        predict(
          md1(),
          make_future_dataframe(
            md1(),
            periods = 6, freq = "month"
          )
        )
      )
    } else if (input$forecastmodel == "autoarima"){
      autoplot(md2())
    } else if (input$forecastmodel == "tbats"){
      autoplot(md3())
    } else if (input$forecastmodel == "auto"){
      autoplot(md4())
    } else if (input$forecastmodel == "hw"){
      autoplot(md5())
    }
  })
  
  # Affichage des résultats de prévision dans une table
  output$forecastdata <- renderDT({
    
    if (input$forecastmodel == "fbpro"){
      predict(
        md1(),
        make_future_dataframe(
          md1(),
          periods = input$horizon, freq = "month"
        )
      )
    } else if (input$forecastmodel == "autoarima"){
      as.data.frame(forecast(md2(), h = input$horizon))
    } else if (input$forecastmodel == "tbats"){
      as.data.frame(forecast(md3(), h = input$horizon))
    } else if (input$forecastmodel == "auto"){
      as.data.frame(forecast(md4(), h = input$horizon))
    } else if (input$forecastmodel == "hw"){
      as.data.frame(forecast(md5(), h = input$horizon))
    }
    
  })
  
}
```


Le code suivant définit la logique serveur (`server.R`) pour une application R Shiny qui effectue des prévisions sur les ventes de produits pharmaceutiques. Voici une explication détaillée pour des débutants.

- Importation des bibliothèques

```r
library(shiny)
library(dplyr)
library(plotly)
library(tidyr)
library(DT)
library(TSstudio)
library(lubridate)
library(forecast)
library(prophet)
library(shinythemes)
```
Ces lignes chargent les bibliothèques nécessaires pour créer l'application. Chaque bibliothèque a des fonctionnalités spécifiques :
- `shiny` : pour créer des applications web interactives.
- `dplyr` : pour manipuler les données.
- `plotly` : pour créer des graphiques interactifs.
- `tidyr` : pour réorganiser les données.
- `DT` : pour créer des tableaux interactifs.
- `TSstudio` : pour l'analyse des séries temporelles.
- `lubridate` : pour travailler avec les dates.
- `forecast` et `prophet` : pour les modèles de prévision des séries temporelles.
- `shinythemes` : pour ajouter des thèmes à l'application Shiny.

- Importation des données

```r
productdb <- readRDS("productdb.rds")
```
Cette ligne charge les données des produits pharmaceutiques à partir d'un fichier `RDS` (format de fichier R pour le stockage des objets R).

- Préparation des données

```r
productdb2 <- productdb %>%
  group_by(BNFNAME, CHEMSUB) %>%
  summarise(SUMQ = sum(QUANTITY)) %>%
  arrange(desc(SUMQ))
```
Ce bloc de code regroupe les données par nom de produit (`BNFNAME`) et par catégorie chimique (`CHEMSUB`), puis calcule la somme des quantités (`SUMQ`) pour chaque groupe. Ensuite, il trie les données par `SUMQ` dans l'ordre décroissant.

- Fonction pour trouver les 5 premiers produits

```r
getTop5 <- function(x) {
  result <- productdb2 %>%
    filter(CHEMSUB == x) %>%
    head(5) %>%
    pull(BNFNAME)
  return(result)
}
```
Cette fonction prend une catégorie (`x`) et renvoie les noms des cinq produits les plus prescrits dans cette catégorie.

- Définition de la logique serveur

```r
function(input, output, session) {
```
Cette ligne définit la fonction principale qui contient toute la logique serveur de l'application.

- Données réactives pour les 5 meilleurs produits

```r
getSalesData <- reactive({
  if(input$forecastmetric == "Quantity") {
    productdb %>%
      filter(BNFNAME %in% getTop5(input$categ)) %>%
      select(month, BNFNAME, QUANTITY) %>%
      rename(Month = month, Product = BNFNAME, Metric = QUANTITY)
  } else if(input$forecastmetric == "Sales Revenue") {
    productdb %>%
      filter(BNFNAME %in% getTop5(input$categ)) %>%
      select(month, BNFNAME, ACTCOST) %>%
      rename(Month = month, Product = BNFNAME, Metric = ACTCOST)
  }
})
```
Cette fonction réactive crée une dataframe avec les données des cinq meilleurs produits dans la catégorie sélectionnée par l'utilisateur, en fonction de la métrique choisie (`Quantity` ou `Sales Revenue`).

- Graphique des 5 meilleurs produits

```r
output$top5plot <- renderPlotly({
  plot_ly(
    getSalesData(),
    x = ~ Month,
    y = ~ Metric,
    color = ~ Product,
    type = 'scatter',
    mode = 'lines',
    text = ~ paste("Product: ", Product)
  ) %>%
    layout(title = paste("Top 5 Products in the ", input$categ, "Category"))
})
```
Ce bloc de code crée un graphique interactif montrant les données des cinq meilleurs produits dans la catégorie sélectionnée.

- Données réactives pour le produit sélectionné

```r
getProdData <- reactive({
  if(input$forecastmetric == "Quantity") {
    productdb %>%
      filter(BNFNAME == input$prods) %>%
      mutate(Date = ymd(paste(month, "20", sep = "-"))) %>%
      select(month, QUANTITY, Date) %>%
      rename(Metric = QUANTITY)
  } else if(input$forecastmetric == "Sales Revenue") {
    productdb %>%
      filter(BNFNAME == input$prods) %>%
      mutate(Date = ymd(paste(month, "20", sep = "-"))) %>%
      select(month, ACTCOST, Date) %>%
      rename(Metric = ACTCOST)
  }
})
```
Cette fonction réactive filtre les données pour le produit sélectionné et crée une colonne `Date` au format `yyyy-mm-dd`.

- Série temporelle réactive

```r
ts_data <- reactive({
  ts(
    data = getProdData()$Metric,
    start = c(year(min(getProdData()$Date)), month(min(getProdData()$Date))),
    frequency = 12
  )
})
```
Cette fonction réactive crée un objet de série temporelle (`ts`) pour les données du produit sélectionné.

- Affichage de la série temporelle

```r
output$actual_ts <- renderPlotly({
  if(input$decompose) {
    ts_decompose(ts_data())
  } else {
    ts_plot(ts_data(), title = input$prods)
  }
})
```
Ce bloc de code affiche la série temporelle du produit sélectionné. Si l'option `decompose` est activée, il affiche la décomposition de la série temporelle.

- Préparation des données pour le modèle Prophet

```r
prophet_df <- reactive({
  getProdData() %>%
    select(Date, Metric) %>%
    rename(ds = Date, y = Metric)
})
```
Cette fonction réactive prépare les données pour le modèle Prophet, en renommant les colonnes comme attendu par Prophet (`ds` pour la date et `y` pour la métrique).

- Modèles de prévision

```r
md1 <- reactive({
  prophet(prophet_df())
})

md2 <- reactive({
  forecast(auto.arima(ts_data()))
})

md3 <- reactive({
  forecast(tbats(ts_data()))
})

md4 <- reactive({
  forecast(ts_data())
})

md5 <- reactive({
  forecast(HoltWinters(ts_data()))
})
```
Ces fonctions réactives définissent différents modèles de prévision :
- `md1` : Modèle Prophet.
- `md2` : Modèle Auto ARIMA.
- `md3` : Modèle TBATS.
- `md4` : Modèle ETS.
- `md5` : Modèle Holt-Winters.

- Affichage des prévisions

```r
output$autoplotforecast <- renderPlot({
  if (input$forecastmodel == "fbpro"){
    plot(
      md1(),
      predict(
        md1(),
        make_future_dataframe(
          md1(),
          periods = 6, freq = "month"
        )
      )
    )
  } else if (input$forecastmodel == "autoarima"){
    autoplot(md2())
  } else if (input$forecastmodel == "tbats"){
    autoplot(md3())
  } else if (input$forecastmodel == "auto"){
    autoplot(md4())
  } else if (input$forecastmodel == "hw"){
    autoplot(md5())
  }
})
```
Ce bloc de code affiche les prévisions selon le modèle sélectionné par l'utilisateur. Chaque condition (`if`, `else if`) correspond à un modèle de prévision différent.

- Affichage des résultats de prévision dans un tableau

```r
output$forecastdata <- renderDT({
  if (input$forecastmodel == "fbpro"){
    predict(
      md1(),
      make_future_dataframe(
        md1(),
        periods = input$horizon, freq = "month"
      )
    )
  } else if (input$forecastmodel == "autoarima"){
    as.data.frame(forecast(md2(), h = input$horizon))
  } else if (input$forecastmodel == "tbats"){
    as.data.frame(forecast(md3(), h = input$horizon))
  } else if (input$forecastmodel == "auto"){
    as.data.frame(forecast(md4(), h = input$horizon))
  } else if (input$forecastmodel == "hw"){
    as.data.frame(forecast(md5(), h = input$horizon))
  }
})
```
Ce bloc de code affiche les résultats de prévision dans un tableau interactif (`DT`). Les prévisions sont générées selon le modèle sélectionné et l'horizon de prévision défini par l'utilisateur.

- Conclusion

Ce code `server.R` configure la logique serveur pour une application R Shiny permettant de visualiser et de prévoir les ventes de produits pharmaceutiques. Les utilisateurs peuvent sélectionner des produits, choisir des modèles de prévision, et visualiser les résultats sous forme de graphiques et de tableaux interactifs. Les fonctions réactives garantissent que l'application répond dynamiquement aux sélections et entrées des utilisateurs.