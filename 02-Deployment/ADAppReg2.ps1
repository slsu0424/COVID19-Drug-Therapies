# Set-ExecutionPolicy -Scope Process -ExecutionPolicy  ByPass

# login azure
Write-Host "Step 1 - Logging into Azure..."
    
az Login

# variables
$keyVaultName = "asakeyssuaefdbhdg2dbc4"
$keyVaultSQLUserSecretName = "testsecret01"
$appName = "COVID1_SP1"
#$rgName = "COVID1"

# get info on currently signed-in user
# az ad signed-in-user show

# get all the subscriptions for the logged in account
# az account list --output table

# get current active subscription ID
$subId = az account show --query id --output tsv

echo "Current Subscription ID is": $subId

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
    az keyvault secret set --name $keyVaultSQLUserSecretName --vault-name $keyVaultName --value $spSecret

    