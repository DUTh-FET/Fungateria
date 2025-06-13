----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 12/01/2024 04:56:11 PM
-- Design Name: 
-- Module Name: SPI_Master - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
-- Company: DUTh FET
-- Engineer: Georgios "ChipherZero" Kleitsiotis, MEng Engineering Physics
-- 
-- Create Date: 08.11.2024 12:49:47
-- Design Name: Hardware ADC
-- Module Name: SPI_Master - Behavioral
-- Project Name: Hardware ADC Fungateria
-- Target Devices: Arty A7
-- Tool Versions: 2
-- Description: SPI Controller to interface with DAC peripheral
-- 
-- Dependencies: 
-- 
-- Revision: 1
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

entity SPI_Master is
    generic(
        data_width : integer := 8; --Data width (e.g. 8 bits per SPI frame)
        cpol : std_logic := '0'; -- Clock polarity
        cpha : std_logic:= '0'; -- Clock phase
        clk_div : integer := 8 -- Clock divider for SPI_Clock. Now Generic
    );
    port(
        clk : in std_logic; -- System clock
        reset_n : in std_logic; -- Active low reset
        --clk_div : in integer; -- Clock divider for SPI clock
        enable : in std_logic; -- Enable signal for transaction
        mosi : out std_logic; -- Master Out Slave In
        miso : in std_logic; -- Master In Slave Out
        sclk : out std_logic; -- SPI clock
        ss_n : out std_logic; -- Chip select (Active low)
        tx_data : in std_logic_vector(data_width-1 downto 0); -- Data to transmit
        rx_data : out std_logic_vector(data_width-1 downto 0); -- Received Data
        busy : out std_logic -- Busy signal
    );
end SPI_Master;

architecture behavioral of SPI_Master is

    type SPI_FSM is (idle, load, transfer, finish);
    
    signal state : SPI_FSM := idle;

    signal clk_counter : integer := 0; -- Clock divider counter
    signal sclk_internal : std_logic := '0'; -- Internal SPI clock
    signal bit_counter : integer range 0 to data_width := 0;
    signal tx_buffer : std_logic_vector(data_width-1 downto 0);
    signal rx_buffer : std_logic_vector(data_width-1 downto 0);
    signal toggle : std_logic := '0'; -- Toggle flag for sampling on CPHA

begin
    -- Clock divider to generate SPI Clock (sclk) based on clk_div
    Clock_Divider_Process : process(clk, reset_n)
    begin
        if reset_n = '0' then
            clk_counter <= 0;
            sclk_internal <= cpol;
        elsif rising_edge(clk) then
            if clk_counter >= clk_div then
                clk_counter <= 0;
                sclk_internal <= not sclk_internal;
            else
                clk_counter <= clk_counter + 1;
            end if;
        end if;
    end process;

    -- Assign generated SPI clock (sclk) and Chip Select (ss_n)
    sclk <= sclk_internal;
    ss_n <= not enable;
    -- FSM Based SPI conntrol logic
    SPI_FSM_Process : process (clk, reset_n)
    begin
        if(reset_n = '0') then
            mosi <= '0';
            rx_data <= (others => '0');
            tx_buffer <= (others => '0');
            rx_buffer <= (others => '0');
            bit_counter <= 0;
            busy <= '0';
            state <= idle;
        elsif rising_edge(clk) then
            case state is
                when idle =>
                busy <= '0';
                --report "Idle state";
                if(enable = '1') then
                    state <= load;
                    busy <= '1';
                    --report "should transition to load";
                end if;

                when load =>
                    -- Load tx_data into tx_buffer
                    --report "load state";
                    tx_buffer <= tx_data;
                    bit_counter <= 0;
                    toggle <= cpha;
                    state <= transfer;
                when transfer =>
                    --report "transfer state";
                    -- SPI Transfer process
                    if clk_counter = clk_div then
                        -- Only act on relevant clock phases
                        if toggle = cpha then
                            -- Sample miso on the active edge
                            rx_buffer(bit_counter) <= miso;

                            -- Transmit mosi bit
                            mosi <= tx_buffer(data_width-1);
                            tx_buffer <= tx_buffer(data_width-2 downto 0) & '0'; -- Shift tx buffer register

                            -- Increment bit counter
                            if bit_counter < data_width-1 then
                                bit_counter <= bit_counter +1;
                            else
                                state <= finish; -- Transition to finish state after complete transfer
                            end if;
                        end if;
                        toggle <= not toggle;
                    end if;
                
                when finish =>
                    --report "finish state";
                    -- Latch the received data and end the transaction
                    rx_data <= rx_buffer;
                    busy <= '0';
                    state <= idle;
                end case;
            end if; 
    end process;
end behavioral;
