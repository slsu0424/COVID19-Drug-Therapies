# https://techcommunity.microsoft.com/t5/itops-talk-blog/powershell-basics-how-to-create-an-azure-ad-app-registration/ba-p/811570

# C:\Users\sansu\Documents\GitHub\COVID19-Drug-Therapies\02-Deployment

# sub: 9edd9e25-815c-4cdb-9bc8-d2ba127ec752
# rg: covid2
# tenantid: 

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

Set-ExecutionPolicy -ExecutionPolicy Undefined -Scope CurrentUser

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy ByPass

#Connect-AzAccount


# login azure
Write-Host "Logging into Azure..."
az Login

$subscriptionId = Read-Host "subscription Id"
$resourcegroupName = Read-Host "resource group name"


az account set --subscription $subscriptionId 

$resourceGroup = az group exists -n $resourcegroupName
if ($resourceGroup -eq $false) {
    throw "The Resource group '$resourcegroupName' is not exist`r`n Please check resource name and try it again"
}

Write-Host "Step 1 - Set up App Registration..."

#Install-Module AzureAD

#Connect-AzureAD

#Connect-AzureAD -TenantId $tenantId

az ad app create --display-name covid0 --available-to-other-tenants false

$objectid = Read-Host "Object Id"

Write-Host "Step 2 - Set up App Registration Secret..."
az ad app credential reset --id $objectid --credential-description TestSecret


#$appName = "covid_appReg"
#$appURI = "https://tailwindtraderssalesapp.twtmitt.onmicrosoft.com"
#$appHomePageUrl = "http://www.tailwindtraders.com/"
#$appReplyURLs = @($appURI, $appHomePageURL, "https://localhost:1234")
#if(!($myApp = Get-AzureADApplication -Filter "DisplayName eq '$($appName)'"  -ErrorAction SilentlyContinue))
#{
#    $myApp = New-AzureADApplication -DisplayName $appName -IdentifierUris $appURI -Homepage $appHomePageUrl -ReplyUrls $appReplyURLs    
#}


#2. Add App Key
#$Guid = New-Guid
#$startDate = Get-Date
    
#$PasswordCredential = New-Object -TypeName Microsoft.Open.AzureAD.Model.PasswordCredential
#$PasswordCredential.StartDate = $startDate
#$PasswordCredential.EndDate = $startDate.AddYears(1)
#$PasswordCredential.KeyId = $Guid
#$PasswordCredential.Value = ([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes(($Guid))))