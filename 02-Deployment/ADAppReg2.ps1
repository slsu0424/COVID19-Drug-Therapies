# login azure
Write-Host "Logging into Azure..."
    az Login

# variables
$keyVaultName = "asakeyabcfelaqpgsfnxcy"
$keyVaultSQLUserSecretName = "test01"
$resourceGroupName = "covid2"
$appName = "covid3"

# pass in arguments
#$subscriptionId = Read-Host "subscription Id"
#$resourcegroupName = Read-Host "resource group name"

$subscriptionId = "9edd9e25-815c-4cdb-9bc8-d2ba127ec752"
$rgName = "COVID2"

az account set --subscription $subscriptionId

Write-Host "Step 2 - Create App Registration and Service Principal..."

    $sp_prop = az ad sp create-for-rbac --name $appName --role Contributor

    echo $sp_prop

    # Expected JSON output
    #{
    #    "appId": "generated-app-ID", [client ID]
    #    "displayName": "service-principal-name",
    #    "name": "http://service-principal-uri",
    #    "password": "generated-password", [client secret]
    #    "tenant": "tenant-ID" [tenant ID]
    #}


    # get appId of the service principal
    $spAppId = (($sp_prop) | ConvertFrom-JSON).appId
    echo "App ID": $spAppId

    # get service principal secret
    $spSecret = (($sp_prop) | ConvertFrom-JSON).password
    echo "Secret value": $spSecret

    # get tenant ID
    $spTenant = (($sp_prop) | ConvertFrom-JSON).tenant
    echo "Tenant": $spTenant

Write-Host "Step 3 - Set Key Vault Access Policy..."

    # set permissions for the service principal in Key Vault
    az keyvault set-policy --name $keyVaultName --secret-permissions set delete get list --spn $spAppId

Write-Host "Step 4 - Register Secret in Key Vault..."
    
    # login as the service principal
    az Login --service-principal -u $spAppId -p $spSecret --tenant $spTenant
    
    # set Key Vault key secret
    az keyvault secret set --name $keyVaultSQLUserSecretName --vault-name $keyVaultName --value $spSecret