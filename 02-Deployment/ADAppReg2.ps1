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
#Write-Host "Logging into Azure..."
#az Login

#$subscriptionId = Read-Host "subscription Id"
#$resourcegroupName = Read-Host "resource group name"

Write-Host "Step 1 - Get Azure IDs for the current subscription..."

#$subscriptionId = az account show --query id
#$global:logindomain = az account tenant list
#$global:logindomain = az account show --query tenant-id

#az account tenant list

$subscriptionId = (Get-AzContext).Subscription.Id
$global:logindomain = (Get-AzContext).Tenant.Id

echo "Subscription ID:" $subscriptionId
echo "Azure AD Tenant ID:" $global:logindomain

#Get-AzKeyVault -ResourceGroupName 'covid2'

$keyVaultName = "asakeyabcfelaqpgsfnxcy"
$keyVaultSQLUserSecretName = "test01"


#az account set --subscription "subscription Id" --resource-group "covid2"

#$resourceGroup = az group exists -n $resourcegroupName
#if ($resourceGroup -eq $false) {
#    throw "The Resource group '$resourcegroupName' is not exist`r`n Please check resource name and try it again"
#}

Write-Host "Step 2 - Set up App Registration..."

#az ad app create --display-name covid2 --available-to-other-tenants false

$objectid = ((az ad app create --display-name covid3 --available-to-other-tenants false) | ConvertFrom-JSON).ObjectId

echo "Object ID:" $objectid

Write-Host "Step 3 - Set up App Registration Secret..."

az ad app credential reset --id $objectid --credential-description TestSecret

$arSecretValue = ((az ad app credential reset --id $objectid --credential-description TestSecret) | ConvertFrom-JSON).password

echo "App Registration Secret Value:" $secretValue

Write-Host "Step 4 - Store Secret in Key Vault..."

Write-Information "Step 5 - Register Secret in Key Vault..."
#$secretValue = ConvertTo-SecureString $sqlPassword -AsPlainText -Force
$kvSecret = Set-AzKeyVaultSecret -VaultName $keyVaultName -Name $keyVaultSQLUserSecretName -SecretValue $secretValue



#2. Add App Key
#$Guid = New-Guid
#$startDate = Get-Date
    
#$PasswordCredential = New-Object -TypeName Microsoft.Open.AzureAD.Model.PasswordCredential
#$PasswordCredential.StartDate = $startDate
#$PasswordCredential.EndDate = $startDate.AddYears(1)
#$PasswordCredential.KeyId = $Guid
#$PasswordCredential.Value = ([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes(($Guid))))