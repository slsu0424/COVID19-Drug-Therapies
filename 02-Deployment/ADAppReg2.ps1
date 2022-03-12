# login azure
Write-Host "Step 1 - Logging into Azure..."
    
az Login

# variables
$keyVaultName = "asakeysusaaefdbhdg2dbc4"
$keyVaultSQLUserSecretName = "testsecret01"
$appName = "COVID0_sp1"

# pass in arguments
#$subscriptionId = Read-Host "subscription Id"
#$resourcegroupName = Read-Host "resource group name"

#$subscriptionId = "9edd9e25-815c-4cdb-9bc8-d2ba127ec752"

#$subscriptionId = "[subscription().tenantId]"
#$currAccount = az account list --query "[?contains(IsDefault, 'true')]" --o table

# get all the subscriptions for the logged in account
az account list --output table

# get current active subscriptionId (isDefault = True)
$subId = az account list --query "[?contains(IsDefault, 'true')]"

#$rgName = "COVID1"

echo $subId

az account set --subscription $subId

Write-Host "Step 2 - Create App Registration and Service Principal..."

    $sp_prop = az ad sp create-for-rbac --name $appName --role Contributor

    #echo $sp_prop

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
    echo "Secret Value": $spSecret

    # get tenant ID
    $spTenant = (($sp_prop) | ConvertFrom-JSON).tenant
    echo "Tenant ID": $spTenant

Write-Host "Step 3 - Set Key Vault Access Policy for the Service Principal..."

    # set permissions for the service principal in Key Vault
    az keyvault set-policy --name $keyVaultName --secret-permissions set delete get list --spn $spAppId

Write-Host "Step 4 - Register Service Principal Secret in Key Vault..."
    
    # login as the service principal
    az Login --service-principal -u $spAppId -p $spSecret --tenant $spTenant
    
    # set Key Vault key secret
#    az keyvault secret set --name $keyVaultSQLUserSecretName --vault-name $keyVaultName --value $spSecret

    