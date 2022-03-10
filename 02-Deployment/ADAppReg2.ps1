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

#Connect-AzureAD -TenantId $global:logindomain

#Get-AzKeyVault -ResourceGroupName 'covid2'

$keyVaultName = "asakeyabcfelaqpgsfnxcy"
$keyVaultSQLUserSecretName = "test01"
$resourceGroupName = "covid2"



#az account set --subscription "subscription Id" --resource-group "covid2"

#$resourceGroup = az group exists -n $resourcegroupName
#if ($resourceGroup -eq $false) {
#    throw "The Resource group '$resourcegroupName' is not exist`r`n Please check resource name and try it again"
#}

Write-Host "Step 2 - Create App Registration..."

$userName = ((az ad signed-in-user show -o json) | ConvertFrom-JSON).UserPrincipalName

echo "User Name:" $userName

# declare variables
$appName = "covid1"

# Create app object
$appReg = az ad app create --display-name $appName --available-to-other-tenants false

$objectid = (($appReg) | ConvertFrom-JSON).objectId
$appid = (($appReg) | ConvertFrom-JSON).appId
$displayname = (($appReg) | ConvertFrom-JSON).displayName

echo "Display Name:" $displayname
echo "Object ID:" $objectid
echo "Application (client) ID:" $appid

#$appName = "covid"
#$StartDate = Get-Date
#$EndDate = $StartDate.AddYears(1)

#$appReg = New-AzureADApplication -DisplayName $appName -AvailableToOtherTenants $false

Write-Host "Step 3 - Generate App Registration Secret..."

# Generate Client Secret for App Registration
#az ad app credential reset --id $objectid --credential-description TestSecret

$arSecretValue = ((az ad app credential reset --id $objectid --credential-description TestSecret) | ConvertFrom-JSON).password

# convert plain text password to a secure string
#$secureValue = ConvertTo-SecureString -String $arSecretValue -AsPlainText -Force

#$Password = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($displayname.Secret))

echo "App Registration Client Secret Value:" $arSecretValue
#echo "Password:" $Password
#echo "App Registration Secret Value:" $secureValue

#$appRegSecret = New-AzureADApplicationPasswordCredential -ObjectId $appReg.ObjectId -EndDate $EndDate

$secureSecret = ConvertTo-SecureString -String $arSecretValue -AsPlainText -Force

Write-Information "Setting Key Vault Access Policy"
Set-AzKeyVaultAccessPolicy -ResourceGroupName $resourceGroupName -VaultName $keyVaultName -UserPrincipalName $userName -PermissionsToSecrets set,delete,get,list

#$ws = Get-Workspace $SubscriptionId $ResourceGroupName $WorkspaceName;
#$upid = $ws.identity.principalid
#Set-AzKeyVaultAccessPolicy -ResourceGroupName $resourceGroupName -VaultName $keyVaultName -ObjectId $upid -PermissionsToSecrets set,delete,get,list


Write-Information "Step 4 - Register Secret in Key Vault..."

#$arSecretValue2 = Read-Host "Secret Value"

#$secureValue = ConvertTo-SecureString -String $arSecretValue2 -AsPlainText -Force

#$kvSecret = Set-AzKeyVaultSecret -VaultName $keyVaultName -Name $keyVaultSQLUserSecretName -SecretValue $secureValue
#$secretValue = ConvertTo-SecureString $Password -AsPlainText -Force
#$kvSecret = Set-AzKeyVaultSecret -VaultName $keyVaultName -Name $keyVaultSQLUserSecretName -SecretValue $Password

$kvSecret = Set-AzKeyVaultSecret -VaultName $keyVaultName -SecretValue $secureSecret
