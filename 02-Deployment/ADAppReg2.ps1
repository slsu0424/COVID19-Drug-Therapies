# login azure
Write-Host "Logging into Azure..."
#az Login

# variables
$keyVaultName = "asakeyabcfelaqpgsfnxcy"
$keyVaultSQLUserSecretName = "test01"
$resourceGroupName = "covid2"
$appName = "covid100"

# pass in arguments
$subscriptionId = Read-Host "subscription Id"
$resourcegroupName = Read-Host "resource group name"

Write-Host "Step 1 - Get Azure IDs for the current subscription..."

echo "Subscription ID:" $subscriptionId

Write-Host "Step 2 - Create App Registration..."

# create Application Registration object
$appReg = az ad app create --display-name $appName

echo $appReg

# grab JSON objects
$objectid = (($appReg) | ConvertFrom-JSON).objectId
$clientid = (($appReg) | ConvertFrom-JSON).appId
#$displayname = (($appReg) | ConvertFrom-JSON).displayName

#echo "Display Name:" $displayname
echo "Object ID:" $objectid
echo "Application (client) ID:" $clientid

Write-Host "Step 3 - Generate secret..."

# generate secret for the client App
$arSecretValue = ((az ad app credential reset --id $clientid --credential-description TestSecret) | ConvertFrom-JSON).password
echo "Secret Value:" $arSecretValue

# convert to secure string
$secureSecret = ConvertTo-SecureString -String $arSecretValue -AsPlainText -Force

Write-Host "Step 4 - Create Service Principal..."

# view existing Sp
#az ad sp list

# create AAD service principal - used to grant permissions (role assignments) to the client app
#$spid = (az ad sp create --id $clientid --query objectId  --output tsv)

#az ad sp create-for-rbac --name $clientid

az ad sp create-for-rbac

#az ad sp create --id $clientid


#spid=$(az ad sp list --display-name $clientid --query "[].appId" -o tsv)
#$spid = az ad sp list --id $clientid --query appId -o tsv

#$spid = ((az ad sp list --display-name $clientid) | ConvertFrom-JSON).appId

#echo "sp:" $spid

Write-Host "Step 5 - Set Key Vault Access Policy..."

# set permissions for the service principal
#az keyvault set-policy --name $keyVaultName --secret-permissions set delete get list --object-id $spid

Write-Host "Step 6 - Register Secret in Key Vault..."

#az keyvault secret set --name $keyVaultSQLUserSecretName --vault-name $keyVaultName --value $secureSecret