pipeline {
/*
 * Defining where to run
 */
//// Any:
// agent any

    agent { label 'jenkinsfile' }
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -le
                    # loads modules
                    module purge
                    module load cmake/3.21.2
                    module load gcc/10.2.0
                    module load openmpi/4.1.0-gcc-10.2.0
		    module load mkl/2020.0.166
                    set -x
                    module list
                    mkdir -p build
                    cd build && rm -rf ./*
                    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DUSE_MKL=ON -DBUILD_TEST=ON -DBUILD_PYTHON=OFF ..    
		    # install
                    make install -j
                '''
            }
        }
    }
    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
        failure {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
    }
}
