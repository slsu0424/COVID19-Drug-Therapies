# login azure
# Write-Host "Logging into Azure..."
# az Login

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

# Create app object
$appReg = az ad app create --display-name $appName --available-to-other-tenants false

# grab JSON variables
$objectid = (($appReg) | ConvertFrom-JSON).objectId
$appid = (($appReg) | ConvertFrom-JSON).appId
#$displayname = (($appReg) | ConvertFrom-JSON).displayName

#echo "Display Name:" $displayname
echo "Object ID:" $objectid
echo "Application (client) ID:" $appid

Write-Host "Step 3 - Generate App Registration Secret..."

# Generate Client Secret for App Registration
$arSecretValue = ((az ad app credential reset --id $objectid --credential-description TestSecret) | ConvertFrom-JSON).password
echo "App Registration Client Secret Value:" $arSecretValue

# Convert to Secure string
$secureSecret = ConvertTo-SecureString -String $arSecretValue -AsPlainText -Force

Write-Information "Step 4 - Set Key Vault Access Policy..."

az keyvault set-policy --name $keyVaultName --secret-permissions set delete get list --object-id $objectid

Write-Information "Step 5 - Register Secret in Key Vault..."

az keyvault secret set --vault-name $keyVaultName --name $keyVaultSQLUserSecretName --value $secureSecret