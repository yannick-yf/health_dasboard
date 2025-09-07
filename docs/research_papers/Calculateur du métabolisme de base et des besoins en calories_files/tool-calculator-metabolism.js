$ = jQuery;

$( document ).ready( function() {
    var result_step1;

    $( "#jvmtb-tool-calculator-metabolism .jvmtb-age" ).on( 'input', function() {
        $(this).val( $( this ).val().replace( /[^0-9.,]/g, '' ) );
    } );

    $( "#jvmtb-tool-calculator-metabolism .jvmtb-height" ).on( 'input', function() {
        $(this).val( $( this ).val().replace( /[^0-9.,]/g, '' ) );
    } );

    $( "#jvmtb-tool-calculator-metabolism .jvmtb-weight" ).on( 'input', function() {
        $(this).val( $( this ).val().replace( /[^0-9.,]/g, '' ) );
    } );

    $( "#jvmtb-tool-calculator-metabolism #gender_woman" ).on( 'click touchend', function() {
        $( this ).addClass( 'selected' );
        $( "#jvmtb-tool-calculator-metabolism #gender_man" ).removeClass( 'selected' );
    } );

    $( "#jvmtb-tool-calculator-metabolism #gender_man" ).on( 'click touchend', function() {
        $( this ).addClass( 'selected' );
        $( "#jvmtb-tool-calculator-metabolism #gender_woman" ).removeClass( 'selected' );
    } );

    $( "#jvmtb-tool-calculator-metabolism .jvmtb-equation" ).on( 'change', function() {
        result = null;
        toggle_step_2( false );
        $( "#jvmtb-tool-calculator-metabolism .jvmtb-tool-convertor-result" ).html( '' );

        var equation = $( this ).val();

        if ( equation == 'cu1980' || equation == 'cu1991' || equation == 'tw2014mm' || equation == 'ti2019mm' ) {
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).val( '' );
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).attr( 'placeholder', 'Votre masse maigre' );
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).attr( 'data-body-mass', "false" );
        } else {
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).val( '' );
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).attr( 'placeholder', 'Votre poids' );
            $( '#jvmtb-tool-calculator-metabolism .jvmtb-weight' ).attr( 'data-body-mass', "true" );
        }
    } );

    $( "#jvmtb-tool-calculator-metabolism" ).on( 'submit', function( e ) {
        e.preventDefault();

        var button = $( this ).find( "button[type=submit]" );
        var that = $( this );
        var gender = $( this ).find( ".jvmtb-gender-block .selected .jvmtb-gender" ).val();
        var age = $( this ).find( ".jvmtb-age" ).val();
        var height = $( this ).find( ".jvmtb-height" ).val();
        var weight = $( this ).find( ".jvmtb-weight" ).val();
        var equation = $( this ).find( ".jvmtb-equation option:selected" ).val();
        var body_mass = $( this ).find( ".jvmtb-weight" ).attr( 'data-body-mass' );
        
        button.prop( 'disabled', true );

        $.ajax( {
            method: 'POST',
            url: jvmtb_ajax_url,
            dataType: 'json',
            data: {
                action: 'calculator_metabolism',
                gender: gender,
                age: age,
                height: height,
                weight: weight,
                equation: equation,
                body_mass: body_mass
            },
            success: function( res ) {
                that.find( ".jvmtb-tool-convertor-result" ).html( '<div>' + res.data.result_text + '</div>' );
                result_step1 = res.data.result;
                toggle_step_2( true );
            },
            error: function ( res, status, error ) {
                var err_msg = JSON.parse( res.responseText ).data;
                that.find( ".jvmtb-tool-convertor-result" ).html( '<div>' + err_msg + '</div>' );
            },
            complete: function() {
                button.prop( 'disabled', false );
            }
        } );
    } );

    $( "#jvmtb-tool-calculator-calorie-requirement" ).on( 'submit', function( e ) {
        e.preventDefault();

        var button = $( this ).find( "button[type=submit]" );
        var that = $( this );
        var gender = $( "#jvmtb-tool-calculator-metabolism .jvmtb-gender-block .selected .jvmtb-gender" ).val();
        var age = $( "#jvmtb-tool-calculator-metabolism" ).find( ".jvmtb-age" ).val();
        var height = $( "#jvmtb-tool-calculator-metabolism" ).find( ".jvmtb-height" ).val();
        var weight = $( "#jvmtb-tool-calculator-metabolism" ).find( ".jvmtb-weight" ).val();
        var coef = $( this ).find( ".jvmtb-physical-activity" ).val();
        var physical_activity = $( this ).find( ".jvmtb-physical-activity option:selected" ).val();
        var objective = $( this ).find( ".jvmtb-objective option:selected" ).val();
        
        button.prop( 'disabled', true );

        $.ajax( {
            method: 'POST',
            url: jvmtb_ajax_url,
            dataType: 'json',
            data: {
                action: 'calculator_metabolism_step2',
                gender: gender,
                age: age,
                height: height,
                weight: weight,
                physical_activity: physical_activity,
                coef: coef,
                objective: objective,
                result_step1: result_step1
            },
            success: function( res ) {
                that.find( ".jvmtb-tool-convertor-result" ).html( '<div>' + res.data + '</div>' );
            },
            error: function ( res, status, error ) {
                var err_msg = JSON.parse( res.responseText ).data;
                that.find( ".jvmtb-tool-convertor-result" ).html( '<div>' + err_msg + '</div>' );
            },
            complete: function() {
                button.prop( 'disabled', false );
            }
        } );
    } );

    function toggle_step_2( enable = false ) {
        if ( enable ) {
            $( "#jvmtb-tool-calculator-calorie-requirement .jvmtb-physical-activity" ).attr( 'disabled', false );
            $( "#jvmtb-tool-calculator-calorie-requirement .jvmtb-objective" ).attr( 'disabled', false );
            $( "#jvmtb-tool-calculator-calorie-requirement button[type=submit]" ).attr( 'disabled', false );
        } else {
            $( "#jvmtb-tool-calculator-calorie-requirement .jvmtb-physical-activity" ).attr( 'disabled', true );
            $( "#jvmtb-tool-calculator-calorie-requirement .jvmtb-objective" ).attr( 'disabled', true );
            $( "#jvmtb-tool-calculator-calorie-requirement button[type=submit]" ).attr( 'disabled', true );
            $( "#jvmtb-tool-calculator-calorie-requirement .jvmtb-tool-convertor-result" ).html( '' );
        }
    }
} );